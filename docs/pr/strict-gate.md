# vulkan: COLI_VK_STRICT, an opt-in mode that makes a lost capability fatal

*Refreshed against `upstream/dev` @ b90a369e (2026-10-21). Changes since the
first submission: the CUDA tier compiles `vk_tier.c` a second time, upstream
added two silent `bd_fmt` refusals, and the suite's new timing lines needed
the normalizer extended. All are folded into the branch's rebase.*

## What it does

One helper, `coli_vk_degrade(class, op, reason)`, now reports every point that
would silently leave the GPU off. Three classes:

- **REFUSE** — the device never had the capability (a format, a tile): one
  `[VK] refused:` line per op, never fatal.
- **DEGRADE** — the run *had* it and silently loses it (a shader absent from
  the deployment, a fence wait that failed, the device lost mid-run): one line
  naming the op, and with `COLI_VK_STRICT=1` an abort with exit 70.
- **FALLBACK** — budget or policy: counted, never fatal, no line.

Unset or `0` is byte-for-byte today's behaviour (counters only). The boot line
prints `[VK] strict: on|off` before any early return.

## Why

A deployment whose optional shaders went missing, or whose device dies
mid-run, today degrades to the CPU with at best one `[VK]` line and at worst
nothing at all — `coli_vk_mark_lost()` was silent. We hit exactly this while
benchmarking: a RADV hard recovery mid-prefill printed one line and the run
continued on the CPU, quietly turning a GPU measurement into a CPU one. The
gate sites were chosen so strict stays safe on Lavapipe, small-shared-memory
GPUs and f32 checkpoints: capability probes are refusals, not degradations.

## Where it gates

- The absent optional shaders/pipelines at boot (`qmatmul_gemm.spv`,
  `qmatmul_coop.spv`, `qmatmul_gate_up.spv`, `rmsnorm.spv`,
  `attention_absorb.spv`), rejected tiles, and the core shader/pipeline.
- The fence/submit waits of every hot op: matmul, gate_up, expert_group_take
  and _take2, pw_flush, attention_absorb and _project, attn_qprep, the expert
  batch submit/join, xb_sub_join, xb_step_sum, `up_lost`.
- Device loss (`coli_vk_mark_lost`, `_dev`), the tier's tensor-pad format
  refusals, the tier's three format refusals at boot, and **new at b90a369e**:
  the two `bd_fmt` early returns in `coli_vk_matmul`/`coli_vk_matmul_pair`
  (a GGUF's block-format dense matrix has no per-op shader — without the line
  a whole GGUF deployment can compute every dense matmul on the CPU and print
  nothing).

## CUDA builds (new at b90a369e)

`cu_tier.c` compiles `vk_tier.c` again behind ~57 macro redirects. The gate
has no CUDA side by design (it is a Vulkan-deployment feature); the tier's
three `coli_vk_degrade` calls compile to a no-op there via one macro in
`cu_tier.c`'s table, so CUDA-only and `VK=1 CUDA=1` links stay green and CUDA
behaviour is byte-identical. The REFUSE counter is not kept in CUDA builds —
the refusal lines themselves still print with `[CUDA]` via `VKT_TAG`.

## Tests

`tests/test_vk_degrade.c` (no device) runs the class table; `make
vk-strict-check` runs the fault knobs against a real device:
default-untouched (unset ≡ 0, stderr-normalized), the refusal is named and
not fatal, exactly one abort on the deliberate final fault, strict-green on
Lavapipe, exactly one boot line. The stderr normalizer now also masks
`N us` lines — upstream's bigger `test_vk_tier` suite prints them and they
broke the unset≡0 comparison.

## Compatibility

Default-off, byte-identical default behaviour, no new env knob surface beyond
the one documented `COLI_VK_STRICT` row. The two committed `.spv` blobs the
original branch carried are dropped — upstream builds them from
`shaders/blockdec.glsl` at b90a369e.
