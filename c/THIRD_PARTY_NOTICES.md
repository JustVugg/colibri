# Third-party notices

Portions of `backend_cuda_dsv4.cu`, including the DeepSeek-V4 GPU router
selection algorithm, are adapted from `ds4_cuda.cu` in the ds4 project:
https://github.com/antirez/ds4

MIT License

Copyright (c) 2026 The ds4.c authors
Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

`tests/bench_dsv4_flashinfer.cu` instantiates the FlashInfer SM120 grouped
MXFP4 GEMM interface from https://github.com/flashinfer-ai/flashinfer.

Copyright (c) 2023-2026 FlashInfer authors

Licensed under the Apache License, Version 2.0. You may obtain a copy at
https://www.apache.org/licenses/LICENSE-2.0.

The DeepSeek-V4 DeepGEMM oracle and benchmark tools follow the FP8 packing
and kernel-dispatch interfaces used by the vLLM project:
https://github.com/vllm-project/vllm

The pinned DeepSeek-V4 reference runner in `tools/dsv4_vllm_reference.py`
executes vLLM commit `ffd46bfab2128bb84146050e98b51a617c6575ab`
without replacing or modifying its kernels. It is retained as the behavioral
and performance oracle for the native Colibri port.

`backend_cuda_dsv4_mhc_vllm.cu` is the CUDA source generated from that exact
commit's TileLang mHC decode kernels for the DeepSeek-V4-Flash configuration
(`hc=4`, `hidden=4096`, `n_out=24`, `split_k=8`). It is kept as the pinned
upstream source for the native ABI integration.

`backend_cuda_dsv4_qkv_vllm.cu` adapts that commit's
`fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu`. The Torch registration
layer is replaced by a native C ABI; the CUDA Q RMSNorm, RoPE, UE8M0 FP8
quantization, and paged KV-cache insertion algorithm remains upstream code.

Copyright contributors to the vLLM project

Licensed under the Apache License, Version 2.0. You may obtain a copy at
https://www.apache.org/licenses/LICENSE-2.0.
