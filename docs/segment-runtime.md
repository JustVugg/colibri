# Layer-segment runtime ABI

`c/segment_runtime.h` is Colibri's engine-neutral boundary for callers that
execute a contiguous, half-open layer range (`begin <= layer < end`). It is a
local C ABI, not a network protocol. A distributed caller remains responsible
for peer identity, transport, leases, request IDs, placement and retry policy.

The ABI is deliberately separate from every model's internal structs:

- Colibri owns weights, kernels, accelerator selection and sequence state.
- An adapter opens only the requested range and exposes the resulting state
  schema and numeric compatibility class.
- A caller creates an isolated session for each conversation, sends boundary
  activations through `coli_segment_run`, and can stream snapshots for
  migration or recovery.

No engine adapter is registered by this initial API change, so existing CLI,
server and model behavior is unchanged. Adapters are added and validated per
model family in later changes.

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

## Run contract

Input and output contain exactly `rows * state_width` values in the advertised
dtype. Token IDs are either absent or contain one entry per row; an adapter can
make them mandatory with `COLI_SEGMENT_CAP_TOKEN_IDS`. The runtime validates
sizes and context bounds before calling the adapter.

Positions and model-specific ordering rules remain adapter-owned. A failed or
cancelled run must not be reported as committed by the distributed caller.
Network-level idempotency and duplicate request handling belong above this ABI.

## Snapshot contract

Snapshot callbacks stream bytes so neither side needs a second full-state
allocation. The format is private to an adapter and compatible only when the
model identity, `state_schema`, numeric class and segment range match. A network
service should put those fields in its own snapshot envelope before accepting a
restore.

## All-family conformance gate

`tests/test_segment_conformance` keeps the ABI universal before model adapters
are connected. It registers six deterministic, stateful fixtures matching all
families in `family_registry.py`:

| Family | Remote state represented by the fixture |
| --- | --- |
| GLM-5.2 | MLA latent cache, RoPE, DSA indexer and device cache |
| Inkling | global/sliding KV and convolutional state |
| Kimi K3 | MLA, KDA recurrent state, convolution windows and AttnRes |
| OLMoE | conventional key/value cache |
| Qwen3.6 | attention KV, DeltaNet recurrent state and convolution ring |
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
release gate is all-or-nothing across all six families: a passing synthetic
fixture alone must never be advertised as model support.

`tests/segment_conformance_manifest.json` binds this matrix to the authoritative
family registry. Adding a future Colibri family without adding its Segment
state and oracle entry fails the Python suite. Five current families have a
generated tiny checkpoint; OLMoE's existing oracle uses its real checkpoint and
is labelled accordingly rather than being presented as a tiny-model result.
