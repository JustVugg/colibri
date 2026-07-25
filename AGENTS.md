# AGENTS.md — Colibri QLoRA Trainer for 64 GB Apple Silicon

> Mission: build an experimental, memory-bounded QLoRA/SFT training path for GLM-5.2 using Colibri's streamed int4 model format on a **64 GB Apple Silicon Mac**, while preserving Colibri's existing inference correctness and dependency-light runtime.
>
> Target repository: https://github.com/JustVugg/colibri
>
> Status of this document: implementation brief for coding agents. Treat every milestone as gated by tests. Do not skip directly to full-model training.

---

## 1. Primary objective

Build a trainer that can fine-tune **small LoRA adapters** against the frozen, quantized GLM-5.2 base model without loading the full ~370 GB routed-expert set into unified memory.

The trainer must exploit the same core property that makes Colibri inference possible:

- keep dense/shared model state resident;
- keep most routed experts on local NVMe;
- load only experts needed by the current training work;
- never allocate gradients or optimizer state for frozen base weights;
- update only explicitly selected LoRA adapter matrices;
- aggressively recompute/checkpoint activations instead of retaining the whole forward graph.

This is **not** conventional full-model QLoRA where the complete quantized checkpoint is resident in RAM/VRAM.

The first usable target is supervised fine-tuning (next-token cross-entropy), not RLHF, PPO, DPO, GRPO, distillation, or full-weight training.

---

## 2. Hard hardware target

Primary machine:

- Apple Silicon Mac
- 64 GB unified memory
- fast local NVMe storage
- macOS
- Metal available

### Memory policy

The trainer must operate under a strict memory budget.

Default targets:

```text
Physical unified memory:       64 GB
Recommended trainer RSS goal: <= 52 GB
Hard safety ceiling:          <= 56 GB
OS / filesystem / safety:      8+ GB reserved
Swap during steady training:   unacceptable
```

Do not claim success if macOS is continuously swapping.

Memory usage must be measured and logged. A run that technically survives by generating heavy swap traffic is a failed 64 GB implementation.

### Initial training defaults

Start conservatively:

```text
micro_batch_size = 1
sequence_length  = 128 or 256
lora_rank        = 4 or 8
grad_accum       = configurable
activation checkpointing = on
base weights     = frozen
router           = frozen
MTP training     = off
routed-expert LoRA = off
```

Increase sequence length only after memory accounting is proven.

---

## 3. Repository facts agents must respect

At the time this brief was written, Colibri already provides:

- a faithful GLM-5.2 `glm_moe_dsa` inference implementation;
- a streamed-expert architecture where dense/shared state stays resident and routed experts live mostly on disk;
- an int4 model format;
- a per-layer expert cache / pinning system;
- a current Apple Silicon Metal backend in `c/backend_metal.mm`;
- Metal kernels for quantized GEMV and routed-expert SwiGLU paths;
- a `METAL=1` macOS build path;
- correctness tests and a token-exact tiny-model oracle workflow.

Relevant current files include:

```text
c/glm.c
c/backend_metal.mm
c/backend_metal.h
c/tier.h
c/st.h
c/decode_batch.h
c/tests/
c/tools/
```

Do not create a parallel model loader until the existing loader and tensor metadata have been evaluated for reuse.

Do not replace or destabilize the default inference path to make training easier.

---

## 4. Critical architectural constraint: do NOT LoRA every routed expert in v1

GLM-5.x has 256 routed experts per MoE layer and 75 MoE layers in Colibri's current layout, i.e. roughly 19,200 routed experts before considering auxiliary/MTP structures.

Attaching rank-8 LoRA to gate/up/down projections of every expert would itself create billions of trainable adapter parameters and enormous gradient/optimizer state. That defeats the 64 GB goal even though the base weights remain quantized and frozen.

Therefore:

### V1 allowed LoRA targets

Use a strict allowlist of **resident dense modules**, starting with attention projections only.

Suggested progression:

1. output projection only;
2. selected attention projections;
3. all approved dense attention projections;
4. optional first dense MLP layers / shared expert only after profiling.

### V1 forbidden targets

Do not train:

- all routed experts;
- router weights;
- embeddings;
- LM head;
- MTP head;
- base quantized weights;
- every layer indiscriminately without a measured adapter-state budget.

### Important naming warning

GLM-5.2 already contains architectural low-rank q/kv projections as part of MLA/DSA. These are **base-model architecture**, not PEFT LoRA adapters.

Do not confuse native `q_lora_rank` / `kv_lora_rank` model structure with the new trainable LoRA adapter layers.

---

## 5. Required high-level architecture

Use this conceptual design:

```text
Dataset / tokenizer
        |
        v
Micro-batch tokens
        |
        v
Colibri forward using frozen int4 base
        |
        +--> resident dense tensors
        |
        +--> router chooses sparse experts
        |        |
        |        v
        |   stream/cache needed experts from NVMe
        |
        +--> LoRA residual on allowlisted dense projections
        |
        v
Cross-entropy loss
        |
        v
Backward / recompute
        |
        +--> no dW for frozen base
        +--> compute dX through frozen W^T
        +--> compute dA/dB only for LoRA
        |
        v
AdamW or memory-conscious optimizer
        |
        v
adapter checkpoint
        |
        v
Colibri inference with base + adapter
```

The core QLoRA equation for an adapted linear layer is:

```text
y = Q(W) x + scale * B(Ax)
```

where:

- `Q(W)` is the frozen quantized base matrix;
- `A` and `B` are small trainable matrices;
- gradients never update `Q(W)`;
- backward still requires multiplication by the transpose of the frozen base matrix to propagate `dX`.

---

## 6. Preferred implementation strategy

### Keep inference stable

Do not turn `c/glm.c` into a monolithic training engine in the first patch.

Prefer a staged structure such as:

```text
c/
  glm.c                       # existing inference entry/path
  backend_metal.mm            # existing Metal backend
  backend_metal.h
  lora.h                      # shared adapter definitions/runtime
  lora.c                      # adapter loader + CPU reference ops
  train/
    train_main.c              # trainer CLI entry
    train_model.c             # training forward/backward orchestration
    train_model.h
    autograd_min.c            # explicit/manual gradient helpers if needed
    optimizer.c
    optimizer.h
    checkpoint.c
    checkpoint.h
    dataset.c
    dataset.h
  tests/
    test_lora.c
    test_train_linear.c
    test_train_tiny.c
    test_adapter_io.c
```

Exact filenames may change if repository conventions suggest a better fit.

The default inference executable must continue to build without training dependencies.

### Avoid introducing PyTorch as a runtime dependency

Python/PyTorch/Transformers may be used as:

- an oracle;
- a converter;
- a test harness;
- a reference gradient implementation.

They should not be required for the final native inference runtime.

For the training path, a Python prototype is acceptable during validation, but the 64 GB streamed full-model path must not require loading a complete Hugging Face checkpoint into PyTorch memory.

### MLX policy

MLX may be used for experiments or reference kernels, but do not assume `mlx-lm` is a drop-in trainer for Colibri's GLM-5.2 streamed representation.

The preferred production direction is to reuse Colibri's existing storage/tiering and Metal mechanisms, adding the minimum training-specific backward/update support needed.

---

## 7. Adapter format: implement before training

Before implementing full backward, make Colibri able to load and apply an adapter during inference.

### Required adapter metadata

Store at least:

```json
{
  "format": "colibri-lora-v1",
  "base_model": "GLM-5.2",
  "base_fingerprint": "...",
  "rank": 8,
  "alpha": 16,
  "dtype": "f16-or-f32",
  "targets": ["..."],
  "tensor_name_map_version": 1
}
```

Adapter tensor names must map deterministically to Colibri layer/tensor names.

### Requirements

- Never require merging adapters into the ~370 GB base model.
- Reject adapters whose base fingerprint / architecture metadata does not match unless an explicit unsafe override is provided.
- Allow adapter load/unload without modifying base files.
- Support an inference smoke test proving that enabling the adapter changes logits/output.
- Include a zero-adapter test proving numerical identity with the unadapted path.

### Suggested CLI shape

```bash
COLI_MODEL=/models/glm52_i4 \
COLI_ADAPTER=/adapters/my_adapter \
COLI_METAL=1 \
./coli chat
```

or an equivalent explicit flag:

```bash
./coli chat --model /models/glm52_i4 --adapter /adapters/my_adapter
```

Follow existing CLI conventions rather than inventing incompatible argument parsing.

---

## 8. Training forward/backward design

Do not build a generic autograd engine unless proven necessary.

GLM-5.2 has a fixed known architecture. A manual backward path can be more memory predictable and easier to checkpoint.

### Frozen linear backward

For:

```text
y_base = Q(W) x
```

base weights are frozen, so only compute:

```text
dx += Q(W)^T dy
```

Never allocate `dW`.

### LoRA backward

For:

```text
z      = A x
y_lora = s * B z
```

compute:

```text
dB += s * dy * z^T
dz  = s * B^T * dy
dA += dz * x^T
dx += A^T * dz
```

Use accumulation precision validated by tests. Start with f32 accumulators even if adapter storage is f16/bf16-like.

### Quantized transpose multiplication

A major required kernel is:

```text
dx = dequant(Q(W))^T * dy
```

Do not materialize a full dequantized matrix.

Implement a streaming/blocked transpose multiply that reads packed int4 + scales directly.

CPU reference first. Metal optimization second.

### Other backward operators required

Implement and test gradients for the operations actually traversed by selected LoRA targets, including as needed:

- RMSNorm;
- residual add;
- SiLU / SwiGLU;
- attention softmax;
- RoPE-related transformations where gradients pass through;
- MLA/DSA projection path;
- selected sparse routing path handling;
- cross-entropy loss.

Router weights remain frozen in v1. Routing decisions are treated as discrete/frozen selection for the step; do not attempt gradient-through-top-k routing in the first implementation.

---

## 9. Activation checkpointing is mandatory

A conventional retained forward graph will violate the 64 GB target.

Default strategy:

1. retain only minimal checkpoint tensors at transformer-block boundaries;
2. store token IDs, positions, routing choices, and compact metadata needed to replay a block;
3. during backward, recompute the block forward;
4. load/reload required frozen experts as needed;
5. compute gradients for the block's adapters and input activation;
6. release block temporaries immediately.

Pseudo-flow:

```text
FORWARD
  for layer in 0..N:
      save minimal checkpoint input
      save routing IDs / routing weights if required for deterministic replay
      run layer
  compute loss

BACKWARD
  for layer in N..0:
      reload checkpoint input
      reload/recompute exact required frozen tensors/experts
      recompute forward intermediates for this layer
      backward layer
      accumulate LoRA gradients
      free intermediates
```

Deterministic replay matters. The recompute path must use the same routing choices and numerically compatible kernels as the original forward.

---

## 10. MoE streaming rules for training

Training creates much greater expert diversity than single-token decode. Naively processing a long sequence can touch a large fraction of experts.

Agents must explicitly optimize expert I/O.

### Required strategies

Investigate and benchmark:

- grouping tokens by selected expert within a layer;
- deduplicating expert loads across all tokens in the micro-batch;
- processing experts in blocks;
- retaining hot experts inside the configured RAM budget;
- overlapping NVMe reads with Metal compute;
- storing/reusing routing decisions for backward;
- deciding whether backward should reload or temporarily retain an expert based on measured memory/I/O cost.

### Never do this

Do not loop like:

```text
for token:
  for selected_expert:
    load expert from disk
```

when the same expert can be loaded once and applied to all routed rows in the current layer.

Use layer-wise expert batching.

### Training cache policy

Inference cache heuristics may not be optimal for training.

Implement a training-specific budget with metrics:

```text
expert_cache_hit_rate
expert_bytes_read_per_step
unique_experts_per_layer
unique_experts_per_step
NVMe read time
Metal compute time
CPU compute time
recompute time
```

Do not hide these metrics.

---

## 11. 64 GB memory budget model

Every major allocation must be categorized and reported.

At startup and periodically during training, log approximately:

```text
resident base tensors
KV / attention scratch (training should avoid inference-only excess)
activation checkpoints
recompute scratch
expert cache/pin
LoRA parameters
LoRA gradients
optimizer state
dataset/token buffers
Metal buffers
miscellaneous RSS
```

Use a central memory budget manager rather than scattered magic limits.

### Dynamic budget rule

Given a requested total budget, reserve first for:

1. mandatory dense/base tensors;
2. adapter + gradient + optimizer state;
3. maximum training scratch;
4. activation checkpoint reserve;
5. OS safety margin.

Only the remainder may be used for expert pin/cache.

The expert cache must shrink before the process is allowed to enter swap pressure.

### Suggested initial command

```bash
./coli train \
  --model /models/glm52_i4 \
  --data ./data \
  --adapter-out ./adapters/run1 \
  --ram 52 \
  --seq-len 128 \
  --micro-batch 1 \
  --grad-accum 16 \
  --lora-rank 8 \
  --lora-alpha 16 \
  --lora-target attention \
  --checkpoint every-layer \
  --metal
```

This CLI is aspirational; adapt it to existing Colibri conventions.

---

## 12. Metal implementation rules

Colibri already has an experimental Metal backend. Extend it instead of starting a second unrelated Apple GPU layer.

Current Metal code already demonstrates:

- packed quantized GEMV;
- int4/int8/f32 paths;
- routed-expert batched GEMV/SwiGLU;
- shared/unified-memory-oriented buffer handling;
- command-buffer batching;
- overlap concepts between expert I/O and GPU work.

### Add training kernels incrementally

Suggested order:

1. f32/f16 small LoRA GEMM/GEMV;
2. LoRA gradient kernels (`dA`, `dB`, `dx_lora`);
3. quantized transpose multiply for frozen base `dx`;
4. RMSNorm backward;
5. SwiGLU backward;
6. attention/softmax backward;
7. fused kernels only after correctness is established.

### Rule: correctness before fusion

Every Metal kernel must have a CPU reference test.

Do not fuse operations until standalone kernels match reference tolerances.

### Avoid excessive dispatch overhead

Follow the existing Metal backend design principle: batch work into a small number of command buffers where possible.

Do not issue one tiny GPU dispatch for every LoRA matrix if CPU execution is faster at that shape. Benchmark and select thresholds.

---

## 13. Optimizer

Start with AdamW for correctness, but make optimizer memory explicit.

For each trainable adapter parameter, account for:

- parameter storage;
- gradient;
- first moment;
- second moment;
- optional master copy.

### Requirements

- no optimizer state for frozen base parameters;
- support gradient accumulation;
- support gradient clipping;
- save/restore optimizer state for resume;
- detect NaN/Inf;
- configurable learning rate and schedule;
- deterministic small-test mode.

If AdamW state is too large for expanded targets, implement a lower-memory optimizer only as a separately validated option. Do not silently change optimizer semantics.

---

## 14. Dataset and tokenizer

Reuse the model's official tokenizer/chat template behavior where possible during preprocessing.

The native trainer should consume a simple pre-tokenized format first to keep model training separate from tokenizer complexity.

Suggested first format:

```text
train.bin / train.idx
valid.bin / valid.idx
metadata.json
```

or a simple line-oriented tokenized representation.

A Python preprocessing tool may convert JSONL/chat datasets into native token IDs.

### Initial SFT semantics

Support:

- causal next-token loss;
- optional prompt masking;
- attention mask / sequence boundaries;
- deterministic shuffling with seed;
- validation split;
- packing only after correctness is proven.

Do not begin with dynamic multi-sequence packing if it complicates routing/backward validation.

---

## 15. Checkpoint and resume

A training checkpoint must include:

```text
adapter weights
adapter metadata
optimizer state
step number
epoch/data cursor
RNG state
scheduler state
training configuration
base model fingerprint
```

Use atomic writes:

```text
write temp -> fsync where appropriate -> rename
```

A crash must not destroy the last valid adapter checkpoint.

### Resume acceptance test

Run:

```text
A: 20 uninterrupted steps
B: 10 steps -> save -> restart -> 10 steps
```

Under deterministic test settings, final adapter tensors and losses should match within defined tolerance.

---

## 16. Required implementation milestones

Do these in order.

### Milestone 0 — Baseline preservation

Before changes:

- build current repository;
- run `make check`;
- on Apple Silicon, build `METAL=1` and run `metal-test`;
- record baseline tiny-oracle behavior;
- record current inference RSS and output on a known prompt if a real model is available.

No training work proceeds if the baseline is already broken.

### Milestone 1 — LoRA adapter inference

Implement:

- adapter structures;
- adapter file format;
- loader;
- name mapping;
- CPU LoRA residual application;
- optional Metal LoRA application;
- zero-adapter identity test;
- nonzero adapter logit-difference test.

Definition of done:

- existing inference without adapter is unchanged;
- adapter-enabled inference works;
- adapter files are small and independent of base model files.

### Milestone 2 — Tiny linear QLoRA primitive

Create a toy frozen int4 linear layer + trainable LoRA.

Implement:

- forward;
- CE or MSE toy loss;
- frozen-base transpose backward;
- LoRA backward;
- optimizer step.

Compare gradients against a high-precision Python reference.

Do not continue until gradient errors are understood.

### Milestone 3 — Tiny transformer training oracle

Use the repository's tiny GLM architecture/oracle approach.

Create a tiny model that fits fully in memory and compare:

- forward logits;
- loss;
- adapter gradients;
- one optimizer step;
- several-step loss trajectory

against Transformers/PyTorch or another trusted reference implementation.

Target adapters only on a small allowlisted set.

### Milestone 4 — Full block backward with checkpointing

Implement backward for all operators needed to propagate through the model while updating only allowed adapters.

Prove:

- full-block gradient parity on tiny model;
- activation checkpoint/recompute parity;
- bounded memory independent of number of retained forward intermediates.

### Milestone 5 — Streamed expert backward/recompute

Add real Colibri expert streaming to the training path.

Requirements:

- layer-wise deduplicated expert loads;
- routing decisions replayed in backward;
- frozen experts never receive gradients;
- no full-expert-set residency;
- expert I/O metrics emitted.

Validate first with a tiny synthetic MoE snapshot.

### Milestone 6 — Metal training acceleration

Port hotspots only after CPU correctness:

- quantized transpose `dX`;
- LoRA forward/backward;
- attention and MLP backward hotspots;
- batched expert recompute/backward-input operations.

Run CPU-vs-Metal parity tests.

### Milestone 7 — Real GLM-5.2 64 GB smoke test

Run with:

```text
seq_len = 128
micro_batch = 1
rank = 4 or 8
attention-only LoRA
small dataset
few training steps
```

Success criteria:

- no OOM;
- no sustained swap;
- RSS stays below configured hard ceiling;
- forward/backward completes;
- loss is finite;
- adapter updates are nonzero;
- checkpoint can be loaded by Colibri inference.

### Milestone 8 — Overfit test

Use a tiny repeated dataset and prove the trainer can deliberately overfit.

Expected evidence:

- training loss decreases materially;
- adapter norm changes;
- model output moves toward target completion;
- disabling adapter returns base behavior.

### Milestone 9 — Practical training improvements

Only after correctness:

- longer sequence lengths;
- better expert scheduling;
- improved cache policy;
- prefetch;
- I/O/compute overlap;
- mixed precision adapter state;
- gradient accumulation performance;
- dataset packing;
- selective layer targeting.

---

## 17. Tests that must exist

At minimum add tests for:

```text
adapter serialization round trip
adapter base-model mismatch rejection
zero LoRA identity
LoRA forward CPU reference
LoRA dA gradient
LoRA dB gradient
LoRA dx gradient
quantized frozen W^T backward
RMSNorm backward
SwiGLU backward
attention/softmax backward
optimizer single step
checkpoint resume
routing replay determinism
expert deduplication per layer
memory budget rejection
Metal vs CPU LoRA parity
Metal vs CPU quantized-transpose parity
tiny end-to-end loss decrease
```

### Gradient test methodology

Use both:

- analytic comparison against PyTorch/reference autograd;
- finite differences for small isolated kernels.

Document tolerances per dtype/kernel.

Do not use overly loose tolerances merely to make tests pass.

---

## 18. Preserve Colibri's existing validation culture

Before every substantial merge:

```bash
make check
```

For Apple Metal changes:

```bash
cd c
make glm METAL=1
make metal-test
```

Also preserve the repository's tiny-model token oracle validation.

The current contributing guidance expects clean builds and oracle validation; training changes must not weaken those gates.

Any change to shared forward kernels must prove inference behavior is unchanged when adapters/training are disabled.

---

## 19. Instrumentation requirements

Every training step should be able to report:

```text
step
loss
learning_rate
tokens_processed
step_time
tokens_per_second
forward_time
backward_time
optimizer_time
recompute_time
NVMe_read_time
NVMe_bytes_read
unique_experts_loaded
expert_cache_hit_rate
RSS
peak_RSS
estimated_adapter_memory
estimated_optimizer_memory
```

On macOS, include best-effort detection/logging for memory pressure and swap activity.

Do not print misleading throughput that excludes expert I/O or backward.

Distinguish:

```text
forward tok/s
training tok/s
wall-clock step time
```

---

## 20. Performance priorities

Order of optimization:

1. stay within 64 GB without swap;
2. correct gradients;
3. minimize unique expert loads per layer;
4. overlap disk I/O with compute;
5. accelerate frozen transpose matmuls;
6. accelerate LoRA gradients;
7. reduce command-buffer overhead;
8. tune cache/pinning;
9. increase sequence length.

Do not optimize superficial Python/CLI overhead while NVMe reads dominate the step.

---

## 21. Expected bottlenecks and research risks

Agents must be honest about these risks.

### Expert fan-out across sequences

A training sequence can route different tokens to many different experts. The union of selected experts can approach a substantial fraction of experts in a layer.

This can make training vastly more I/O-heavy than autoregressive inference.

### Backward doubles/repeats base access

Frozen weights still participate in gradient propagation through `W^T`. Checkpointing may require rereading/recomputing base tensors.

### Dense adapter scope may limit fine-tune capacity

Attention-only LoRA is chosen for memory feasibility, not because it is guaranteed to match full expert adaptation quality.

### Metal kernels are inference-oriented today

Existing Metal support is valuable, but training requires new transpose/backward operators and careful command scheduling.

### Training speed may be very slow

The success criterion for v1 is **feasible, correct fine-tuning on 64 GB**, not GPU-cluster-class throughput.

Do not promise a particular tokens/sec before measurement on target hardware.

---

## 22. Optional future: sparse expert LoRA

Do not implement until dense-adapter training works.

A future design may adapt only a bounded subset of routed experts:

```text
--expert-lora-policy none
--expert-lora-policy top-N-hot
--expert-lora-policy explicit-list
--expert-lora-budget-mb N
```

Possible strategy:

- choose a small hot expert set from usage statistics;
- allocate LoRA only for those experts;
- all other experts remain completely frozen/no-adapter;
- enforce a hard adapter + optimizer memory budget.

Never implicitly create adapters for all 19k+ experts.

Any sparse-expert adapter format must encode exactly which layer/expert IDs are adapted.

---

## 23. Non-goals for first release

Do not block v1 on:

- LoRA for every routed expert;
- distributed training;
- multi-Mac training;
- full-parameter training;
- RLHF / PPO / GRPO;
- MTP-head training;
- router training;
- 1M-token context training;
- multi-user server training;
- automatic Hugging Face adapter compatibility for every PEFT naming convention;
- Windows/Linux training parity.

Focus on one target: **a correct 64 GB Apple Silicon streamed QLoRA/SFT proof of concept with a useful adapter output**.

---

## 24. Suggested CLI surface

Target user experience eventually:

```bash
# Build
cd c
make glm METAL=1
make train METAL=1
make metal-test

# Preprocess data
python3 tools/prepare_sft.py \
  --input data.jsonl \
  --output data/tokenized \
  --model zai-org/GLM-5.2

# Train
COLI_MODEL=/Volumes/FastNVMe/glm52_i4 \
COLI_METAL=1 \
./coli train \
  --data data/tokenized \
  --output adapters/my-run \
  --ram 52 \
  --seq-len 128 \
  --micro-batch 1 \
  --grad-accum 16 \
  --rank 8 \
  --alpha 16 \
  --targets attention \
  --lr 1e-4 \
  --steps 100

# Validate adapter
COLI_MODEL=/Volumes/FastNVMe/glm52_i4 \
COLI_ADAPTER=adapters/my-run \
COLI_METAL=1 \
./coli chat
```

The implementation may use a separate `coli_train` executable initially. Integration into the `coli` wrapper should happen only after the training binary is stable.

---

## 25. Agent workflow rules

When working on this project:

1. Read the current repository before editing; it is evolving quickly.
2. Work against the repository's integration branch policy, not assumptions from stale documentation.
3. Keep patches small and milestone-scoped.
4. Add tests in the same patch as new math/kernels.
5. Never change quantization semantics silently.
6. Never silently fall back to a different precision during training.
7. Never silently exceed the configured RAM budget.
8. Never report success based only on a toy model when claiming 64 GB GLM-5.2 support.
9. Never benchmark from a warm filesystem cache and call it cold NVMe throughput.
10. Preserve the default dependency-free inference build.
11. Do not merge large architectural refactors without first proving a minimal vertical slice.
12. Prefer measurable experiments over speculative optimization.

### Before coding a milestone

Write down:

```text
Goal
Files expected to change
Memory impact
Numerical risk
Tests to add
Benchmark to run
Rollback/fallback behavior
```

### After coding a milestone

Report:

```text
What changed
Tests passed
Gradient/reference error
RSS / peak RSS
Disk bytes per step
Step time breakdown
Known limitations
Next blocker
```

---

## 26. First vertical slice agents should implement

The smallest meaningful end-to-end slice is:

```text
1. Add LoRA adapter structure for ONE dense projection.
2. Apply it in CPU inference.
3. Add adapter save/load.
4. Build a toy int4 frozen linear + LoRA trainer.
5. Validate dA/dB/dX against PyTorch.
6. Train toy data until loss falls.
7. Add same LoRA path to Metal.
8. Validate CPU vs Metal.
9. Extend to one tiny GLM attention projection.
10. Validate tiny-model forward + gradients.
```

Do not start by attempting a real 744B training step.

---

## 27. Definition of done for the 64 GB proof of concept

A milestone may be called **64 GB QLoRA POC complete** only when all of the following are demonstrated on a real 64 GB Apple Silicon Mac:

- GLM-5.2 Colibri int4 base model is used from local NVMe;
- full base model is not resident in unified memory;
- selected LoRA adapters are trainable;
- a real forward + backward + optimizer step completes;
- at least 100 consecutive training steps complete on a small SFT dataset;
- peak RSS remains under the declared safety ceiling;
- steady-state swap is not used as hidden model memory;
- loss remains finite and shows expected learning behavior;
- checkpoints resume correctly;
- resulting adapter loads in Colibri inference;
- disabling the adapter restores base behavior;
- `make check` still passes;
- Metal correctness tests pass where Metal is used;
- performance and I/O numbers are reported honestly.

A useful stretch definition of done:

- tiny dataset can be intentionally overfit;
- training throughput is stable after warm-up;
- expert cache hit rate and bytes-read/step are measurable;
- no unexplained memory growth over a long run.

---

## 28. Reference facts and sources

Use current upstream sources when implementation details conflict with this brief.

- Colibri repository: https://github.com/JustVugg/colibri
- Colibri README / architecture and Metal notes: https://github.com/JustVugg/colibri/blob/main/README.md
- Colibri contributing/testing rules: https://github.com/JustVugg/colibri/blob/main/CONTRIBUTING.md
- Colibri Metal backend: https://github.com/JustVugg/colibri/blob/main/c/backend_metal.mm
- Colibri C Makefile / `METAL=1`: https://github.com/JustVugg/colibri/blob/main/c/Makefile
- Hugging Face Transformers GLM MoE DSA docs: https://huggingface.co/docs/transformers/model_doc/glm_moe_dsa
- Official GLM-5.2 model page: https://huggingface.co/zai-org/GLM-5.2
- MLX-LM LoRA/QLoRA reference documentation: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md
- QLoRA paper: https://arxiv.org/abs/2305.14314

---

## 29. Final engineering principle

The project succeeds by treating **memory capacity, SSD storage, and Apple GPU compute as one managed hierarchy** while keeping training state tiny.

The central invariant is:

```text
Never make the 744B base trainable.
Never require all routed experts in memory.
Never let adapter/optimizer state scale accidentally with all 19k+ experts.
Recompute instead of retaining when memory is the limiting resource.
Batch expert access by layer to minimize NVMe traffic.
Prove correctness on tiny models before scaling.
```

Build the smallest correct path first. Measure everything. Optimize only what the measurements show is dominant.
