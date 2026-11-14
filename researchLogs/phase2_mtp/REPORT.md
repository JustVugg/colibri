# Phase 2: MTP Activation, Metal Integration and Decode Performance

## Summary

The supplied Qwen3.6-27B IQ2_XXS GGUF does not run MTP. It advertises one
next-token-prediction layer in metadata, but the inspected 866-tensor inventory
contains no MTP head tensors, and the checkpoint is dense while the current MTP
loader requires MoE/shared-expert tensors. A real invocation with `Q36_MTP=1`
confirmed `[mtp] Q36_MTP=1: the head stays off`. Consequently, draft, acceptance,
verification and MTP-versus-Metal measurements are unavailable for this model;
the one-draft, adaptive-depth and deeper-draft benchmark cases are blocked, not
zero-performance MTP results.

Three matched 64-token greedy CPU/Metal control trials ran against the real
model. All six output streams were byte-identical. Metal IQ2_XXS GEMV was
invoked 13,986 times in each Metal trial, with no counted CPU fallback; the CPU
trials counted 13,986 CPU IQ2_XXS matmuls and no Metal dispatches. The median
post-first-token rate was 1.185 tokens/s for Metal versus 0.705 tokens/s for
CPU (derived from the existing run duration and TTFT measurements). This is
about 68% higher for Metal on this workload, but the repeated results varied,
and the full-request rate was dominated by the long prompt prefill.

## Runtime architecture and execution path

The model loader calls `q36_mtp_attach(&m)` after the trunk/device setup. The
attach path only attempts loading when `Q36_MTP=1`; `q36_mtp_load()` requires
`mtp.fc.weight` and additional MTP norm, attention, router and expert tensors.
It also rejects model geometry without experts/shared-expert support. The
inspected GGUF metadata says `qwen35.nextn_predict_layers = 1`, but its tensor
inventory has 866 tensors and no `mtp.*` names. The runtime smoke invocation
with MTP requested reported that the head stayed off. There is therefore no
attached head, no readiness state, no adaptive gate decision, and no MTP draft
or verification for this checkpoint.

For a compatible checkpoint, the intended runtime path is:

1. `q36_mtp_attach()` conditionally loads and initializes the head.
2. `q36_spec_begin()` enables MTP only when the head is attached and the active
   CUDA-tier, cache-route and qpack restrictions permit speculation. Prompt
   lookup competes with MTP unless disabled with `COLI_LOOKUP=0`.
3. Before drafting, `q36_spec_step()` checks `q36_mtp_ready(pos)`. Fixed depth
   is controlled by `Q36_MTP_DRAFTS` (default one); `auto` selects depth through
   the speculative gate. Probability and periodic plain-step conditions can
   reduce deeper fixed drafts.
4. `q36_mtp_draft()` and `q36_mtp_draft_more()` calculate draft rows in the MTP
   implementation. On the Metal-only build, the MTP head computations are
   CPU-side; compatible one-row IQ2_XXS projections can call the native Metal
   GEMV helper through `matmul_d()`.
5. `step_ex()` verifies the accepted draft batch and the model state is rolled
   back to the accepted prefix when rows are rejected. The native IQ2_XXS
   GEMV helper requires a single row (`S == 1`); thus batched projections do
   not use it, while eligible row-wise projections may.
6. `q36_mtp_feed()` feeds model outputs into the MTP head/KV state for later
   drafts.

The ordinary dense model path did use the Phase 1 native Metal GEMV when
enabled. No MTP execution-path claims are inferred from the metadata flag.

## Instrumentation

Opt-in instrumentation is enabled with `Q36_PERF_STATS=1`. With it unset or
zero, counters do not increment and no Phase 2 report is printed. The
instrumentation reports:

- MTP proposals, verified drafts, accepted tokens, acceptance among verified
  drafts, and average accepted tokens per MTP verification.
- Speculative verify forwards/rows and target verification row throughput,
  ordinary decode forwards, draft elapsed time and proposals/second.
- MTP head rows/time and feed K/V rows/time.
- IQ2_XXS Metal GEMV and CPU fallback counts separated by decode, MTP feed,
  MTP draft, verification and other phases.
- Existing CLI TTFT, total-run token rate and peak RSS.

For the supplied model, the controls report zero drafts, zero verification
forwards, zero MTP head/feed rows, and 63 ordinary decode forwards per 64-token
run. The initial prompt prefill is not included in the Phase 2 dispatch counters
or the 63-forward count. The existing CLI `Speed` is the complete `generate()`
duration, including prompt prefill; a post-first-token throughput is derived
below from `64-token run duration - TTFT`, and is not presented as an
independently timed kernel metric. Because these are CLI runs with
`NOSTREAM=1`, they do not measure server, transport, terminal, or UI delivery
latency. Target verification throughput and draft-generation throughput are
not applicable while MTP is inactive.

## Environment and benchmark method

- Branch/revision: `feature-IQ2_XXS`, `c92e16ed`.
- Machine: Apple M4, 16 GiB unified memory; macOS 27.2 (build 26B123).
- Model: `/Users/creemac/.lmstudio/models/unsloth/Qwen3.6-27B-MTP-GGUF/Qwen3.6-27B-UD-IQ2_XXS.gguf`.
- Model shape from runtime: 64 layers, 16 attention layers, hidden size 5120,
  MLP size 17408, vocabulary 248320; dense, no experts.
- Prompt: `logs/phase2_mtp/benchmark_prompt.txt`, tokenized as 1,536 tokens.
- Generation: 64 tokens, greedy CLI generation, identical input/model/settings.
- Executable: built by `make -C c qwen36-phase2-stats-check qwen36 METAL=1`.
  CPU trials use that same binary with `COLI_METAL_IQ2_XXS=0`; Metal trials use
  `COLI_METAL_IQ2_XXS=1`.
- Common environment: `SNAP=<model path> N_NEW=64 Q36_PERF_STATS=1 Q36_MTP=0
  COLI_LOOKUP=0 NOSTREAM=1`, invoked as
  `./c/qwen36 0 4 logs/phase2_mtp/benchmark_prompt.txt`.
- Three sequential pairs were run, with CPU before Metal in each pair. The
  trial order was not randomized. Logs and stdout are retained as
  `cpu_1..3.{stderr,stdout}.log` and `metal_1..3.{stderr,stdout}.log`.
- Peak RSS is the engine's reported peak RSS; it is not the model file size.

The literal common command was run once for each mode and trial, changing only
the mode flag and output log names:

```sh
SNAP=<model path> N_NEW=64 Q36_PERF_STATS=1 Q36_MTP=0 COLI_LOOKUP=0 \
COLI_METAL_IQ2_XXS=<0-or-1> NOSTREAM=1 \
./c/qwen36 0 4 logs/phase2_mtp/benchmark_prompt.txt
```

The prompt and output are long enough to expose repeated decode steps, but
prefill is unusually expensive for this model and dominates full-run latency.
The short historical Phase 1 GEMV tests are not compared as full generation
benchmarks.

## Repeated control results

Times are median (range) across three trials. “Post-TTFT output rate” is
`63 / (reported total generate seconds - reported TTFT)`; it covers the
remaining 63 generated tokens after the first token. The full-request rate
includes prefill and is based on the CLI's printed `Speed`.

| Metric | CPU, IQ2 Metal disabled | Metal IQ2_XXS |
|---|---:|---:|
| TTFT | 801.067 s (730.677–849.961) | 855.026 s (819.972–868.389) |
| Total `generate()` duration | 890.4 s (843.3–937.5) | 908.2 s (877.3–921.2) |
| Full-request throughput (CLI) | 0.07 tok/s (0.07–0.08) | 0.07 tok/s (0.07–0.07) |
| Post-TTFT output throughput (derived) | 0.705 tok/s (0.559–0.720) | 1.185 tok/s (1.099–1.193) |
| Peak RSS | 8.82 GB (8.72–8.84) | 5.46 GB (4.79–5.60) |
| IQ2_XXS GEMV / fallback by decode phase | 0 / 13,986 | 13,986 / 0 |
| MTP proposed / verified / accepted | 0 / 0 / 0 | 0 / 0 / 0 |
| Ordinary decode forwards | 63 each | 63 each |

The post-TTFT throughput ratio is approximately 1.68x in favor of Metal. The
full-request throughput does not show that improvement because TTFT/prefill
dominates. This distinction is essential: the CLI's full-run token rate is
user-request latency including prompt processing, while the derived rate
isolates output after first token; neither is target-model verification
throughput or draft-generation throughput.

## MTP benchmark cases

| Requested case | Result |
|---|---|
| CPU baseline, MTP disabled | Completed, three trials |
| Metal IQ2_XXS, MTP disabled | Completed, three trials |
| Metal, one MTP draft | Blocked: no loadable MTP head |
| Metal, adaptive MTP depth | Blocked: no loadable MTP head |
| Metal, deeper fixed MTP drafts | Blocked: no loadable MTP head |

Enabling `Q36_MTP=1` in the real-model smoke run did not attach the head. No
draft acceptance percentage or MTP draft/verification throughput is applicable;
reporting these as measured zeros would be misleading.

## Correctness and tests

- CPU versus Metal greedy output: all six stdout files have the same SHA-256,
  `51c7887b72272da13bd956aeaffdaf1bd3c58fbcd07ea89e7017138c658fcb9d`.
- Repeated-run determinism: three CPU and three Metal outputs were identical
  byte-for-byte.
- `make -C c qwen36-phase2-stats-check qwen36 METAL=1`: passed; the focused
  opt-in/phase-separated counter test passed and the Metal engine built.
- From `c/`, `./tests/test_qwen36_mtp_depth`: passed.
- From `c/`, `./tests/test_qwen36_eos`: passed.
- From `c/`, `./tests/test_spec_decode_state`: passed (the test intentionally
  prints an oracle non-finite-logit diagnostic while exercising its state
  handling).
- `git diff --check`: passed.

Greedy output identity is the available CPU/Metal behavioral check. This
checkpoint cannot exercise MTP-vs-disabled output, speculative logit/argmax
agreement, rejection rollback in a real MTP run, MTP EOS/stop handling, or
termination partway through an MTP verification batch. Those correctness
checks remain unverified, not implicitly passed.

## Bottleneck and recommendation

The measured full request is dominated by prompt prefill: TTFT is 731–868
seconds versus 843–938 seconds total. Metal materially improves the
post-first-token decode rate and removes all counted IQ2_XXS CPU fallback calls
for this workload, but end-to-end request rate remains around 0.07 tokens/s at
the tested 1,536-token prompt. These measurements do not identify MTP head,
verification, KV, expert routing, synchronization, or allocation costs because
the head is inactive and the model is dense.

The next optimization to investigate is long-prompt prefill, because it
dominates measured end-to-end latency. Profile its per-layer compute and
dispatch before changing it; these runs do not establish whether prefill time
is dominated by IQ2_XXS arithmetic, attention, setup, or synchronization. Also
obtain a Qwen3.6 checkpoint whose tensor inventory and architecture meet
`q36_mtp_load()` requirements, then rerun the same controlled CPU/Metal
controls and add MTP modes with the exact same checkpoint and prompt. Only
after phase counters show MTP drafts and verification rows should profiling
target MTP head compute, row-wise versus batched GEMV dispatch, command-buffer
submission, activation/scratch reuse, and KV work. Do not optimize those
unmeasured paths based on this report.

## Risks and untested cases

- A metadata `nextn_predict_layers` value does not guarantee that the matching
  MTP tensors or supported architecture are present.
- The trial order was fixed CPU-then-Metal, rather than randomized; thermal,
  cache and background-load effects may contribute to the observed range.
- The prompt produces an unusual visible continuation; it is a controlled
  fixed input, not a representative chat-quality evaluation.
- There are only three trials per control and no compatible MTP checkpoint.
- No logit-tensor comparison, real MTP rejection/rollback check, stop-sequence
  test under active MTP, or end-of-generation partial verification case was
  possible.
- Metal GEMV counters count successful helper invocations; they do not
  attribute total time to the Metal device or quantify synchronization cost.
- Model placement, available memory and system load can change RSS and
  throughput; the reported RSS values are observations from this machine and
  run sequence only.
