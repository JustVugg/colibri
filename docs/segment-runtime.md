# Layer-segment runtime ABI

`c/segment_runtime.h` is Colibri's engine-neutral boundary for callers that
execute a contiguous, half-open layer range (`begin <= layer < end`). It is a
local C ABI, not a network protocol. A distributed caller remains responsible
for peer identity, transport, leases, request IDs, placement and retry policy.
The companion [`edge_runtime.h`](../c/edge_runtime.h) supplies the model-owned
tokenizer, embedding and final head needed to drive a complete Segment chain.

The ABI is deliberately separate from every model's internal structs:

- Colibri owns weights, kernels, accelerator selection and sequence state.
- An adapter opens only the requested range and exposes the resulting state
  schema and numeric compatibility class.
- A caller creates an isolated session for each conversation, sends boundary
  activations through `coli_segment_run`, and can stream snapshots for
  migration or recovery.

All seven model families have engine-owned CPU adapters, but no ordinary CLI or
server links or registers them. A Segment host opts in by linking the dedicated
adapter objects and calling the explicit registration functions. Existing
standalone initialization and inference therefore remain unchanged.

## Lifecycle

1. Call each model adapter's explicit registration function during process
   initialization. The consumer must not depend on linker constructors; this
   keeps initialization order visible and portable to MSVC.
2. Open an engine with `coli_segment_engine_open` and inspect its model-specific
   `ColiSegmentCapabilities`.
3. Create one or more sessions. A session must not receive concurrent calls.
4. Run, snapshot or restore each session as needed.
5. Destroy every session, then close the engine.

Engine close fails while a session is alive. Registration must finish before
concurrent lookups begin; registration itself is not a hot-path operation.

## Capability identity

Capabilities are returned after model open because the layer count, boundary
width and context limit may vary between checkpoints handled by one engine.
The caller initializes `struct_size` to its allocation size. The runtime zeros
that complete allocation before copying the fields it knows, so a future caller
using a larger structure never observes uninitialized extension fields when it
loads an older runtime.
`state_schema` identifies the activation and snapshot layout.
`numeric_class` identifies builds whose results and snapshots are compatible;
an adapter must include every relevant precision, reduction and backend rule in
that class.

`COLI_SEGMENT_CAP_RANGE_NATIVE` is a strong promise: the adapter did not load
weights outside the requested range. Callers must not publish range-native
residency when this bit is absent.

## Real adapters

`c/segment_adapters.h` exposes explicit registration for GLM-5.2, Inkling,
Kimi K3, OLMoE, Qwen3.6, Qwen3.8-Flash-Next and DeepSeek V4. The adapters retain model weights in
the engine and conversation state in isolated sessions:

| Adapter | Boundary/state contract |
| --- | --- |
| GLM-5.2 | hidden activations; MLA latent and DSA index caches |
| Inkling | hidden activations; global/sliding-ring KV and four conv rings |
| Kimi K3 | hidden plus every AttnRes block residual; KDA, conv, MLA and DSA |
| OLMoE | hidden activations and conventional KV |
| Qwen3.6 | hidden activations; attention KV, DeltaNet recurrent and conv state |
| Qwen3.8-Flash-Next | four-stream hyper-residual activations; QSA KV/indexer, GDN recurrent/conv and PLE hash history |
| DeepSeek V4 | expanded `hc_mult * hidden` mHC state; window/compressed attention, compressor and indexer |

The current adapter build advertises CPU only. This is intentional capability
truthfulness, not a limitation of the ABI: GPU flags will be added per engine
only when the corresponding Colibri backend is executed by the adapter.
`make -C c segment-adapters` builds all seven together and verifies that their
identities register in one runtime. They are never pulled into `colibri`,
`inkling`, `kimi_k3`, `olmoe`, `qwen36`, `qwen38` or `deepseek_v4` by that target.

## Run contract

Input and output contain exactly `rows * state_width` values in the advertised
dtype. Token IDs are either absent or contain one entry per row; an adapter can
make them mandatory with `COLI_SEGMENT_CAP_TOKEN_IDS`. The runtime validates
sizes and context bounds before calling the adapter.

Positions and model-specific ordering rules remain adapter-owned. A failed or
cancelled run must not be reported as committed by the distributed caller.
Network-level idempotency and duplicate request handling belong above this ABI.
Cancellation is cooperative, not asynchronous: an adapter may check only at a
model-safe boundary, and Qwen3.8 currently checks before entering a complete
Segment run. Callers that need prompt-time interruption should terminate or
migrate the worker and retry from the last published snapshot.

## Snapshot contract

Snapshot callbacks stream bytes so neither side needs a second full-state
allocation. The format is private to an adapter and compatible only when the
model identity, `state_schema`, numeric class and segment range match. A network
service should put those fields in its own snapshot envelope before accepting a
restore.

## All-family conformance gate

`tests/test_segment_conformance` keeps the ABI universal independently of
model files. It registers seven deterministic, stateful fixtures matching all
families in `family_registry.py`:

| Family | Remote state represented by the fixture |
| --- | --- |
| GLM-5.2 | MLA latent cache, RoPE, DSA indexer and device cache |
| Inkling | global/sliding KV and convolutional state |
| Kimi K3 | MLA, KDA recurrent state, convolution windows and AttnRes |
| OLMoE | conventional key/value cache |
| Qwen3.6 | attention KV, DeltaNet recurrent state and convolution ring |
| Qwen3.8-Flash-Next | four-stream hyper-residual boundary, QSA/indexer KV, GDN recurrence and PLE history |
| DeepSeek V4 | mHC, window/compressed attention, compressor and indexer |

Every fixture must pass the same checks for half-open range identity, exact
activation geometry, session isolation, contiguous execution, streamed
snapshot/restore, exact continuation and transactional rejection of corrupt or
range-incompatible snapshots. The test is dependency-free and runs on every
platform in the ordinary C and sanitizer suites.

The fixture schemas are prefixed with `fixture/`. They exercise the contract;
they are not model math and are never registered by a shipping executable. A
model is ready for distributed Segment execution only after its real adapter
passes these lifecycle checks against the repository's generated tiny oracle
and the existing token/numerical oracle for that engine. The public Lumabri
release gate is all-or-nothing across all seven families: a passing synthetic
fixture alone must never be advertised as model support.

`tests/segment_conformance_manifest.json` binds this matrix to the authoritative
family registry. Adding a future Colibri family without adding its Segment
state and oracle entry fails the Python suite.

The second gate, `tests/test_segment_adapters_real`, runs actual model math. For
each family it compares one full range with two chained ranges, checks isolated
sessions, continues after snapshot/restore, and proves a corrupt restore is
transactional. Tiny checkpoints come from the existing GLM, Inkling, Kimi,
Qwen and DeepSeek generators plus `tools/make_olmoe_tiny.py`; Qwen and OLMoE
are passed through their production Colibri converters before the test.

Run the complete gate with the seven generated container paths:

```sh
make -C c segment-adapters-real \
  GLM_SEGMENT_MODEL=/path/to/glm_tiny \
  INKLING_SEGMENT_MODEL=/path/to/tiny_inkling \
  KIMI_SEGMENT_MODEL=/path/to/kimi_k3_tiny \
  OLMOE_SEGMENT_MODEL=/path/to/olmoe_merged_tiny \
  QWEN_SEGMENT_MODEL=/path/to/qwen36_converted_tiny \
  QWEN38_SEGMENT_MODEL=/path/to/qwen38_tiny \
  DEEPSEEK_SEGMENT_MODEL=/path/to/deepseek_v4_tiny
```
