# Kolibri-1 (Aleph Alpha)

**Experimental, CPU only.** Aleph Alpha's [Kolibri-1](https://huggingface.co/Aleph-Alpha/Kolibri-1)
(Apache-2.0) runs on `c/kolibri1.c` straight from the FP8 safetensors release, with no
conversion. Qualification on the real checkpoint is incomplete; read
[Correctness status](#correctness-status) before relying on it.

| | Kolibri-1 |
|---|---|
| layers | 50: sliding-window attention on four of every five layers (window 513), full attention (no position encoding) on the fifth |
| routed experts | 384, top-6 per token, `sigmoid_logit_add` router (select on score + bias, weigh by `sigmoid(score)`, no renormalisation) |
| shared expert | one, ungated |
| attention | separate q/k/v/o, per-head q/k RMSNorm |
| on disk | about 78.9 GB (FP8 e4m3 weights with FP32 128x128 scale grids; BF16 embeddings, router, norms, head) |
| context | trained to 262,144; the engine sizes its KV cache from `K1_CTX` / `CTX` |

The checkpoint is read as released. The 7,720 expert tensors are split across shards (about
40 % of the experts are not contiguous, and an expert's scales may sit in another shard
than its weights), so the loader builds a placement table from the shard headers and never
assumes a layout. `./kolibri1 <model> --validate --release` checks all 116,303 tensors
(names, shapes, dtypes, scale grids, byte counts) in about 0.3 s.

## Running it

```sh
cd c
make kolibri1
./kolibri1 /path/to/kolibri-1-fp8 --validate --release
python3 openai_server.py --model /path/to/kolibri-1-fp8 --engine ./kolibri1 --port 8000
coli plan --model /path/to/kolibri-1-fp8 --ram <GB> --ctx <n>      # sizes the expert cache
```

`coli run` has no kolibri1 branch: the engine is driven through the gateway
(`openai_server.py`, `SERVE=1`). Memory: about 74 GiB for every expert of every layer to
stay resident, plus the KV cache, the FP64 workspace and the prefix snapshot (all charged by
the planner); a smaller cache reads experts from disk on demand.

The environment variables are in [ENVIRONMENT.md](ENVIRONMENT.md#kolibri-1-engine-kolibri1).

## What the engine does

- **Loader and tokenizer.** The tokenizer goes through `tok.h`, with a guarded engine-local
  override for the release's one-digit rule (`\p{N}{1}`), which `tok.h` does not detect, so no
  shared header is edited. It is checked against Hugging Face `tokenizers` on the full
  vocabulary, a multilingual corpus and every Unicode scalar value
  (`make kolibri1-tokenizer-real MODEL=...`).
- **Numerics.** FP8 weights stay e4m3 bytes with the checkpoint's grid and are never expanded;
  a block is summed in FP32 and the blocks are added in double after the block scale. The
  residual stream, the RMSNorm, the SiLU-times-up product and the routed-expert accumulation
  run in FP64 (policy `kolibri1-residual-fp64-c1`); activations feeding the FP8 matmuls and the
  head are FP32. A row's result does not depend on the prefill chunk size, the expert cache
  size or the thread count.
- **Gateway.** `render_chat_kolibri1` renders the release's chat template by hand, pinned byte
  for byte against the template itself (all `reasoning_effort` levels, tool history,
  continuation of an open assistant turn). Tool calls are Hermes-style JSON. Sampling defaults
  come from the release's `generation_config.json`. `SERVE=1` adds a per-request cache reset,
  optional prefix reuse by snapshot, cancel, and a finite-logit check that answers with an
  error instead of a token.
- **Sampler.** Temperature, top-p and (launch level) top-k, with the tie rules of the vendor
  stack's sampler, checked by `tests/test_kolibri1_serve_sampler.c`. The random draw uses
  colibri's own RNG, so sampled streams are not identical to another engine's; only the
  support and probabilities of the distribution are the same. `top_k` and the seed are
  launch-level (`K1_TOP_K`, `K1_SEED`); request-level values are ignored with a warning.
- **Planner.** `c/family_registry.py` carries the descriptor, KV/ring/prefix-snapshot
  geometry and resident-weight inventories, so `coli plan` accounts for what the engine
  allocates.

Not supported: images and audio, MTP/speculative decoding, grammar-constrained decoding,
repetition penalty, GPU expert tiers (a CUDA tier exists outside this change).

## How it is tested

**Tiny fixture, in CI (`make -C c kolibri1-check`, numpy only, no torch, no network).**
`tools/make_kolibri1_tiny.py` writes a deterministic random checkpoint in the release layout
(F32 scales first in every shard, a 60/20/20 mix of contiguous and split experts, plus two
variants); `make kolibri1-check` then runs:

- the fixture census and the loader on every fixture, with a negative matrix of malformed
  checkpoints, each refused by name;
- the engine's kernels against an independent NumPy float64 reference;
- the CPU forward pass against the committed `c/kolibri1_tiny/ref.json`: three prompts longer
  than the window, 24 generated tokens each, token-exact, teacher-forced argmax and margin at
  every position, bit-identical across chunk sizes, cache sizes, thread counts and decode vs
  prefill, with a mutation matrix that must be caught;
- the FP64 residual rule against an exact-rational reference;
- the capture hooks and the tokenizer.

Gateway, chat-template, sampling and `SERVE` protocol tests run under `make test-python`
(they skip without the binary, NumPy or jinja2). `make kolibri1-c1-check` additionally compares
the FP8 dequantisation of all 256 codes with torch and is maintainer-run.

**The reference.** `ref.json` is written by `tools/make_kolibri1_ref.py` from a PyTorch
reference implementation vendored in `tools/kolibri1_reference/`. Unlike the MiMo and Kimi
oracles, which run the vendor's own code, this one is a separate implementation: the vendor
stack for Kolibri-1 needs vLLM. It was written from the checkpoint's published semantics, the
vendor plugin and the vLLM sources (not from llama.cpp or any colibri engine),
and was checked against the vendor runtime on tiny models. CI only reads the committed
`ref.json`; regenerating it needs torch.

**Real weights.** The engine was compared against an FP64 PyTorch implementation of the same
checkpoint on a corpus of 2k-, 4k- and 16k-token documents, token for token, layer by layer.
The gates were: KL(reference || engine) mean <= 1e-5, p99 <= 1e-4, max <= 1e-3;
full-vocabulary |logit difference| <= 0.01 (an application tolerance, not a certified
bound); zero top-1 disagreements where the reference's top-1 margin exceeds 0.02, with
at least 95 % of positions decisive; finite values; correct expert selection and weights. The
tooling for that comparison is not part of this repository. The results are in
[Correctness status](#correctness-status).

## Correctness status

> Qualification incomplete; fresh G3 validation FAILED. Frozen CPU candidate C completed all
> declared checks on 99 development cells; 24 additional development cells passed
> output/execution checks with routing deferred. G3 stopped after one of 80 evaluated cells
> exceeded the absolute-logit tolerance (0.0140647 versus 0.01); the other 79 passed the
> evaluated output/execution gates. G3 routing was not evaluated. Evidence for the rebased PR
> binary is limited to the separately listed equivalence and smoke checks.

What that means, in plain terms:

- **Development cells.** Candidate C (the numerics in this change) was run on 123 development
  runs. On 99 of them every declared check passed, including expert routing. On the other 24
  (9 mixed-format 4k runs and 15 16k runs) the output and execution checks passed and the
  routing checks were deferred, not run. Across all 123: KL max 3.7e-9, largest |logit
  difference| 2.1e-3, zero decisive top-1 disagreements.
- **Fresh validation (G3) failed.** G3 was run once on new documents under a sealed
  procedure. It stopped at the first failure: a 16k-token replay run whose full-vocabulary
  logit difference at one prefill position was 0.0140647, over the 0.01 tolerance. The other
  79 evaluated cells passed every evaluated output and execution gate; the routing checks
  of G3 were not evaluated, and one 16k cell was not run. The tolerance was not changed
  afterwards. A diagnostic investigation of the failing cell is in progress and is not a
  qualification.
- **Residual stream.** The scaled-residual difference against the FP64 reference is larger
  than an early 1e-2 diagnostic limit in three development cells (4k mixed 1.8e-2; one 16k
  CSV cell 5.7e-2 at layer 45). Local checks show the engine's arithmetic is correct at those
  points and the map amplifies accumulated rounding (a few hidden coordinates carry very
  large activations). The outputs are within the gates above; the residual is reported, not
  gated.
- **Natural (free-running) runs are not equivalent to the reference over the whole
  population.** Where MoE routing has a near-tie, engine and reference can choose different
  experts and later positions then differ legitimately. Unmasked, with each side routing
  naturally: no top-1 differences on the 2k and 4k development documents; at 16k, 37 of 16,384 positions differ for the English document, 12 for the CSV
  and 6 for the regression prompt, 0 for the German and code documents.
- **Qualified range.** Up to 16,384 positions (15,872 prompt + 512 generated). Longer
  contexts run but are not qualified.
- **J-typical detector: NOT QUALIFIED.** The detector meant to classify routing divergences
  by typicality did not pass its qualification and is excluded from every result above.
- **Unresolved qualification gaps; further work paused.** (F1) routing and envelope identity for
  the nine mixed-format development cells; (F2) routing envelopes for the development 16k
  cells; (F3) routing envelopes for the G3 16k cells; (F4) the detector above. (F5) Limited
  rebased-binary equivalence: the bitwise equivalence of this change's CPU path to the
  evaluated candidate is demonstrated on the tiny fixtures, and one real-weight case is
  compared record for record; broader equivalence on real weights is unestablished.
- **Context, not a threshold.** The official vLLM plugin (locked stack, 58 GB UVA CPU
  offload, offload numerical equivalence unestablished) was compared with the same FP64
  reference on 13 fixed inputs, five fresh sessions per length class. Median per-document mean
  KL ranged from 0.022 to 0.23 for the plugin and from 7.7e-14 to 5.8e-4 for this engine; the
  plugin also differed between fresh sessions by about the same amount. Execution differed
  (GPU prefill chunks of 2,048 with offload against CPU chunks of 64) and the causes are
  unresolved. This describes agreement with the reference, not task quality, and not the
  non-offloaded plugin.

## Known limitations

- Experimental and CPU only; Linux x86-64 is the platform used for real weights. The fixture
  suite builds on macOS arm64. Windows is untested.
- Speed is bounded by memory bandwidth and expert reads. Earlier, unfrozen screens on a
  16-core desktop with the experts resident gave prefill of about 18 s at 600 tokens, 164 s
  at 4,096 and 1,220 s at 15,872, and decode of about 10.3, 8.8 and 4.6 tok/s. There is no
  validated benchmark manifest for this engine yet.
- `top_k` and the seed are launch-level.
- The test hooks compiled into the binary (`K1_CAPTURE`, `K1_PROBE`, `K1_TEST_*`, ...) are off
  by default and print a banner when on.

## Provenance

The checkpoint and Aleph Alpha's plugin are Apache-2.0. `c/tests/fixtures/kolibri1_chat_template.jinja`
is the release's own chat template, kept unmodified to pin the rendered bytes (see
`THIRD_PARTY_NOTICES.md`). No code is taken from llama.cpp, vLLM or another inference engine.
The FP8 e4m3 kernel is duplicated from colibri's own `qwen38_core.h` so the file stands
alone; the sampler and the gateway mirror documented behaviour of the vendor stack and are
checked against it in tests.
