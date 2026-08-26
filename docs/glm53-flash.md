# GLM-5.3-Flash

GLM-5.3-Flash support is tracked in
[#1243](https://github.com/JustVugg/colibri/issues/1243). The official
checkpoint is recognized by `coli plan` and `coli doctor`; inference is not yet
wired, and the launcher refuses it instead of falling back to the GLM-5.2
engine.

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
including MTP. `runtime_available=false` is deliberate until the CPU text path
passes a generated tiny oracle and a real-checkpoint generation test.

## Implementation order

1. Generate a tiny Transformers oracle and pin tensor names, shapes and math.
2. Compose the CPU text path from the existing Kimi KDA and DeepSeek V4
   mHC/DSA primitives; do not copy either full engine.
3. Add FP8-to-streaming conversion and token/logit validation on the real
   checkpoint.
4. Enable the gateway, then add CUDA and Metal tiers without changing router
   semantics.
