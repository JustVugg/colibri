#!/usr/bin/env python3
import os
import torch
from vllm.third_party.deep_gemm import bf16_gemm_nt

o = int(os.environ.get("O", "1024"))
k = int(os.environ.get("K", "4096"))
a = torch.zeros((1, k), dtype=torch.bfloat16, device="cuda")
b = torch.zeros((o, k), dtype=torch.bfloat16, device="cuda")
c = torch.empty((1, o), dtype=torch.bfloat16, device="cuda")
bf16_gemm_nt(a, b, c)
torch.cuda.synchronize()
print("BF16_PROBE", o, k, c[0, 0].item())
