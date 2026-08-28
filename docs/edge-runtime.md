# Model-edge runtime ABI

`c/edge_runtime.h` is Colibri's engine-neutral local C boundary for the two
ends of a distributed inference route:

```text
text -> tokenizer -> embedding -> one or more Segment engines -> final head
```

It is not a network protocol. Colibri owns tokenization, model boundary math,
formats and kernels; a consumer such as Lumabri owns discovery, transport,
placement, leases and session lifecycle. Together with
[`segment_runtime.h`](../c/segment_runtime.h), this lets a chatter process run
a real model without loading every transformer layer locally.

The ABI is additive. Ordinary Colibri CLIs do not link or register Edge
adapters and keep their existing initialization and inference paths.

External consumers can build the complete CPU-baseline runtime with
`make -C c segment-edge-library`. The resulting
`c/build/segment/libcolibri_segment_edge.a` contains both public runtimes and
all six built-in adapters. Consumers link it after their own objects and call
the explicit registration functions below; ordinary Colibri executables do
not link this archive.

## Lifecycle and compatibility

1. Explicitly register the required adapters during process initialization.
   There are no link-time constructors, including on Windows.
2. Open one model boundary with `coli_edge_engine_open`.
3. Read its capabilities and require an exact match with the routed Segment
   chain for `state_schema`, `numeric_class`, state dtype and state width.
4. Tokenize and embed the prompt, run it through the complete Segment chain,
   then call `coli_edge_select` on the last row.
5. For decode, embed the selected token and repeat at the next position.
6. Close the Edge engine after its sessions have drained.

The caller initializes every public structure's `struct_size`. Capability
queries zero the caller's complete allocation before copying fields known to
the runtime, so an older runtime cannot leave future extension fields
uninitialized.

Version 1 deliberately exposes deterministic greedy selection. Sampling is a
future additive operation; it is not emulated outside Colibri. Registration
must finish before concurrent lookup. A consumer should currently host one
active model per chatter process because some existing engine configuration is
process-global; Qwen's production tokenizer makes that restriction explicit
and rejects a second live Qwen Edge engine.

## Real adapters

`c/edge_adapters.h` exposes explicit registration functions for every current
Colibri model family:

| Adapter | Boundary state | Edge residency |
| --- | --- | --- |
| GLM-5.2 | hidden state | tokenizer, embedding, final norm and head |
| Inkling | hidden state | tokenizer, optional embed norm, final norm and head |
| Kimi K3 | hidden plus all AttnRes blocks | tokenizer, final transforms and head; embedding rows stream on demand |
| OLMoE | hidden state | tokenizer, embedding, final norm and head |
| Qwen3.6 | hidden state | tokenizer, embedding, final norm and head |
| DeepSeek V4 | expanded mHC state | tokenizer and small final mHC tensors; BF16 embedding rows and head tiles stream on demand |

Every adapter currently advertises CPU only. Accelerator flags will be added
only when that adapter actually executes its Edge operations on the matching
Colibri backend. `resident_bytes` reports the adapter-owned model tensors and
can be bounded with `memory_limit_bytes`; model files and tokenizer metadata
remain additional mapped/indexed resources.

## API contract

Tokenize and detokenize support a sizing pass by passing a null output with
zero capacity. `coli_edge_embed` accepts exactly one token ID per row and emits
`rows * state_width` values in the advertised dtype. `coli_edge_select`
applies the model's exact final transform and output head to every input row,
returning one greedy token and an optional score per row. Edge ABI v2 also
exposes `coli_edge_logits`, which returns the complete row-major vocabulary
logits after that same model-specific transform. Colibri therefore remains the
owner of model math, while a serving caller can apply temperature, top-p and
its own reproducible RNG policy without duplicating any of the six heads.
`argmax(logits)` is release-gated against `coli_edge_select` for every family.

The runtime validates structure sizes, activation geometry, batch limits,
output capacities and cancellation before entering an adapter. Model adapters
also validate token IDs and checkpoint boundary tensors. Errors are returned
to the caller; the network layer decides whether to retry, migrate or fall back
locally.

## All-family release gate

`make -C c edge-adapters` verifies that all six real adapters coexist in one
runtime. `edge-adapters-real` is the stronger gate: for every family it loads
the real Edge adapter and a full real Segment range, round-trips text, embeds
an independent oracle prompt, performs prefill plus decode and compares three
greedy tokens with that oracle. It also requires generated tokens to be
detokenizable and requires exact Edge/Segment capability identity.
For each family, the gate additionally computes the full logits for the first
decode row and requires their argmax to equal the already oracle-checked
greedy token.

Generate the existing tiny checkpoints from `c/` (they are test data and are
not committed):

```sh
python3 tools/make_glm_oracle.py
python3 tools/make_edge_tiny_tokenizer.py glm_tiny --vocab-size 256

python3 tools/make_tiny_inkling.py tiny_inkling
python3 tools/make_edge_tiny_tokenizer.py tiny_inkling --vocab-size 256

python3 tools/make_kimi_k3_tiny.py --output kimi_k3_tiny --force
python3 tools/make_edge_tiny_tokenizer.py kimi_k3_tiny --vocab-size 320

python3 tools/make_olmoe_tiny.py --output olmoe_tiny_src --force
python3 tools/make_edge_tiny_tokenizer.py olmoe_tiny_src --vocab-size 128
python3 tools/convert_olmoe_merged.py --model olmoe_tiny_src \
  --out olmoe_tiny_merged --min-free-gb 0

python3 tools/make_qwen36_tiny.py --out qwen36_edge_src \
  --emit-ref qwen36_edge_src/ref_qwen36.json --ref-mode full --max-new 8
python3 tools/make_edge_tiny_tokenizer.py qwen36_edge_src --vocab-size 320
python3 tools/convert_qwen36.py --model qwen36_edge_src \
  --out qwen36_edge_i8 --ebits 8 --no-readme

python3 tools/make_deepseek_v4_tiny.py --output deepseek_v4_edge_tiny --force
python3 tools/make_edge_tiny_tokenizer.py deepseek_v4_edge_tiny --vocab-size 128
```

Inkling's generator requires `c/tools/oracle-requirements.txt`. Then run the
single all-family gate:

```sh
make -C c edge-adapters-real \
  GLM_EDGE_MODEL=glm_tiny GLM_EDGE_REF=ref_glm.json \
  INKLING_EDGE_MODEL=tiny_inkling \
  INKLING_EDGE_REF=tiny_inkling/ref_inkling.json \
  KIMI_EDGE_MODEL=kimi_k3_tiny KIMI_EDGE_REF=kimi_k3_tiny/ref.json \
  OLMOE_EDGE_MODEL=olmoe_tiny_merged \
  OLMOE_EDGE_REF=olmoe_tiny_src/ref_olmoe.json \
  QWEN_EDGE_MODEL=qwen36_edge_i8 \
  QWEN_EDGE_REF=qwen36_edge_src/ref_qwen36.json \
  DEEPSEEK_EDGE_MODEL=deepseek_v4_edge_tiny \
  DEEPSEEK_EDGE_REF=deepseek_v4_edge_tiny/ref.json
```

The fixture tokenizer helper only fills the tokenizer deliberately omitted by
math-only generators. Shipping checkpoints always use their production
tokenizer. A family is not considered supported from registration or a
synthetic adapter alone: this oracle gate must pass for all current families.
