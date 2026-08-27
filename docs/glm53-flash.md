# GLM-5.3-Flash

GLM-5.3-Flash support is tracked in
[#1243](https://github.com/JustVugg/colibri/issues/1243). The official
checkpoint is recognized by `coli plan` and `coli doctor`, and the text runtime
is available through `coli run`, `coli chat`, and the OpenAI-compatible API.
The vision tower remains out of scope for this text-only runtime.

## Architecture contract

The text model is not a smaller GLM-5.2. It combines:

- 45 text layers: 34 recurrent KDA and 11 DeepSeek Sparse Attention layers;
- four-stream mHC around attention and FFN blocks;
- three dense FFNs followed by 42 sparse FFNs with 288 routed experts, top-8;
- one additional MTP expert row at layer 45;
- a 128x128 block-FP8 checkpoint under `model.language_model.*`;
- an optional vision tower, outside the first text-only runtime milestone.

The resource planner accounts separately for fixed KDA recurrence/conv state,
context-growing DSA latent/index state, mHC workspace, and all routed experts
including MTP. Routed experts are loaded from their safetensors shard on hit;
dense and shared weights remain resident.

## Validation and acceleration

1. Generate a tiny Transformers oracle and pin tensor names, shapes and math.
   `make -C c glm53-tiny-check` now gates this contract against Transformers
   5.16.1, including cached versus full-forward parity.
2. `tools/check_glm53_checkpoint.py` validates every runtime-consumed tensor
   name, dtype, and shape. It has been run against all 62 official shard
   headers; this proves the load contract, not full-model generation quality.
3. The CPU full-forward and incremental-cache engines match the independent
   tiny Transformers oracle, including native fine-grained FP8 activation QDQ.
4. `COLI_CUDA=1` and `COLI_METAL=1` dispatch native FP8 matrices to the shared
   GPU backends. Both retain GLM-5.3's per-128-element activation QDQ before
   device execution; unsupported builds refuse an explicitly requested backend
   instead of silently running on the CPU.

Build the optional accelerators with:

```sh
make -C c glm53_flash CUDA=1       # Linux CUDA
make -C c glm53_flash METAL=1      # Apple Silicon
make -C c glm53_flash CUDA_DLL=1   # Windows host; build cuda-dll as well
```

The generated oracle is the reproducible correctness gate. End-to-end
generation on the 328 GB official checkpoint and performance measurements on
real CUDA/Metal hardware remain hardware acceptance work; a successful
toolchain-only CI build is not evidence for either claim.
