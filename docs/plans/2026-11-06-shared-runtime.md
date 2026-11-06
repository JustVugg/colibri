# Shared runtime: continuation of RFC #1057

## Baseline and boundaries

Inspected upstream `dev` at `5cfb55bd`, against the maintainer's sequence in
https://github.com/JustVugg/colibri/discussions/1057:
family registry, serving codec, ExpertStore, then Engine/Session adapters.
This is a contributor implementation plan, not an upstream release schedule.

Keep architecture math, tokenizer semantics, routing, KV/recurrent state,
prefix reuse, batching, admission and cancellation policy family-owned.
Do not introduce a generic forward pass or assume all families can restore the
same kind of sequence state.

## Current implementation

| Boundary | Evidence | Remaining work |
| --- | --- | --- |
| Family registry | `c/family_registry.py`, registry tests; #1063 and #1068 | Continue registering concrete capabilities as consumers need them. |
| Shared serving transport | `serve_codec.h` used by OLMoE, Kimi K3, DeepSeek V4 and Inkling; #1087, #1090, #1096, #1116 | GLM mux still owns most framing. Later families need their own coverage audit. |
| ExpertStore | `expert_store.h` defines views and leases; production consumers remain V4-specific | Demonstrate the seam with a second family before designing a common factory. |
| Backend factory | `expert_store_registry.h` accepts V4 engine/config/options | This is a V4 extension seam, not a model-independent registry. |
| Engine/Session | `segment_runtime.h/.c` already provide opaque engine/session handles, live-session close protection and capability checks; real family adapters exist | Reuse this layer-segment runtime, which is separate from ordinary CLI/serve, and audit its capability/lifecycle contracts before broader integration. |

## Ordered delivery

### 1. GLM serving transport, separately reviewable slices

1. Freeze production DATA/ECHO transcripts, then route these frames through the
   shared codec. Preserve numeric IDs through ULLONG_MAX, UTF-8, newlines,
   zero-byte tokens, opt-in logprob tails and ECHO position zero. The test runs
   the production mux emitters with a synthetic tokenizer and no model weights.
2. Freeze SUBMIT/STOP/CANCEL transcripts before moving input framing. The next
   committed baseline covers the real GLM admission/rejection path; shared body
   reads now consume prompt, grammar and the delimiter for both GLM and the
   general command reader. Keep the existing numeric header parser: the general
   reader does not preserve all GLM field combinations or expose `ids=1`. Cover
   six/seven-field headers, numeric request IDs, grammar bytes, pin/logprobs/ids
   extensions, partial payloads, malformed framing and interleaved active slots.
   Keep `ColiSubmit` semantics and queue ownership unchanged.
3. Migrate control responses and completion framing only after pinning GLM's
   exact float formatting, cluster suffix and PROF/DONE flush boundary. Do not
   substitute a sibling's formatter whose wire representation differs.

Acceptance for the entire GLM migration: unchanged golden bytes for valid
frames; explicit failure when malformed input loses synchronization; existing
mux, cancellation, grammar, prefix, logprob and continuous-batching tests pass.
Compile and test siblings after changing shared helpers. A successful first
slice does not complete the migration or establish real-model GPU parity.

### 2. ExpertStore with a second production consumer

The CPU view/lease adapter is proposed separately in #2655. Continue auditing
OLMoE's existing `expert_get`/`expert_put`, busy-slot ownership,
prefetch eviction protection, row-int8 weights, native GGUF blocks and CPU
interleaved rows. Add baseline lease/eviction and output tests before adapting
one path. Do not label repacked bytes as ordinary GGUF tensor views.

A store adapter must preserve the cache's behavior and borrowed weight lifetime:
lookup acquires exactly one lease, release clears it, active leases prevent
replacement, failed lookup exposes no data, and destruction requires no leases.
Keep prefetch policy and GPU cache decisions in the family initially. Preserve
V4 and its existing external backend registration API while the second consumer
establishes which creation options are actually shared.

Acceptance: two real production consumers, failure/ownership tests, identical
CPU results on fixed fixtures, and a measured dispatch-overhead check. GPU paths
require separate hardware acceptance before being claimed migrated.

### 3. Engine/Session adapters

Use the existing `segment_runtime.h/.c` rather than introducing a competing
engine/session framework. It already wraps family adapters, refuses engine close
with live sessions, and exposes explicit snapshot support. Its conformance
fixtures cover ten state topologies; fixture coverage is distinct from real
adapter/model acceptance. Ordinary CLI/serve paths do not use it yet.

Audit capabilities and lifetime rules at this boundary, then establish which
serve responsibilities can reuse it without forcing layer-segment semantics on
full generation. Engine owns weights and stores; sessions retain family-specific
state and must be destroyed first. Store ownership in ordinary family serving
remains separate work. Unsupported prefix/snapshot operations stay explicit.

## Delivered slices

GLM DATA/ECHO emission and SUBMIT body reads use the shared codec. Numeric header
parsing, malformed-header draining, mid-generation control polling, IMAGE,
ACCEPT/ERROR/DONE, scheduling, KV ownership, token decoding and logprob arithmetic
remain family-owned. Each production framing baseline is committed before its
refactor so reviewers can run the same assertions on both revisions.

The intake baseline covers legacy headers, grammar with embedded newlines,
`pin`/`logprobs`/`ids` together, maximum numeric IDs, busy/duplicate/out-of-range
slots, NUL rejection, malformed headers, unknown and malformed control requests,
truncated prompt/grammar, and missing/non-LF delimiters. Rejections that can
continue are followed by another control frame to verify stream alignment.
These synthetic tests do not exercise active-request STOP/CANCEL effects or
model inference. Full header/control migration and hardware acceptance remain
open; this is not a completed GLM migration.
