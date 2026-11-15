# C20 — Disk prompt cache in mux mode (KV_SLOTS>1): scoping note

**Status:** scoped, then **implemented and validated (2026-11-15, PASS — see the C20 section of
`DEV_PHASED_PLAN_TTFT_AND_THROUGHPUT.md`)**. The design below was approved by the user and shipped
as proposed; the validation results supersede the open questions at the bottom.

**Question:** the disk prompt cache (`Q36PC01` snapshot files: per-prompt state
photographs with LRU eviction under a byte/count budget) is deliberately gated
to `g_q36_mux_slots == 1`. The gate's comment calls the feature "a v1
optimisation like every other fast path here". What would it take to let
multiplexed serving (two or more concurrent conversations, `KV_SLOTS=2`) use
the same disk cache — the thing that made cross-restart repeats instant
(0.18 s vs 114 s prefill at 4K) in the single-slot measurements?

## How the pieces actually work today

- **Slot routing (gateway, already done):** the gateway hashes each conversation
  to a stable KV slot (`conversation_cache_slot()`, `c/openai_server.py`), so
  every turn of one conversation lands on the same slot. No gateway change is
  needed.
- **Slot state (engine):** each slot owns a `Q36Seq` — its K/V rows, DeltaNet
  state, `kv_len`, the `kv_prefix` reuse record, an image turn's rope positions,
  ring positions. `q36_seq_swap()` trades the bound conversation's set into the
  Model before anything runs on it; a prefill is therefore *exclusive* — mux
  batches only decode rows (`q36_step_rows`), never prefills.
- **Disk cache touch-points (`q36_pcache_restore` / `q36_pcache_store`):** both
  operate solely on the Model — which, in `serve_mux`, is the bound request's
  slot. Restore writes: K/V rows, `DN_rec`/`DN_conv`, `kv_len`, the `kv_prefix`
  record, and the MTP pending row (`q36_pin_state_put`). Store photographs the
  same set plus the prompt ids and the post-prompt logits row.
- **Where the gate sits:** two `g_q36_mux_slots == 1 &&` conditions inside
  `q36_serve_start` — one at the restore probe (before the live-state reset),
  one at the store call (after the prefill). No other code knows about the gate.

## Feasibility: the touch-points vs the per-slot swap set

| restore/store touches | in the per-slot `Q36Seq` swap? | mux verdict |
|---|---|---|
| K/V rows, `DN_rec`/`DN_conv`, `kv_len`, `kvp`, `mpos`/`ring_pos` | **yes — all of it** | safe: writes land in the bound slot's own state |
| MTP pending row (`g_q36_mtp.pend`) | no (process-global) | dead in mux: `KV_SLOTS>1` disables speculation ("nothing drafts"); the **pin path already touches the same globals under mux today**, so no new exposure |
| device-freshness flags (`dn_gpu_invalidate`, `q36c/cc_host_wrote`) | chain-specific | no-ops on the CPU build; the Vulkan/CUDA mux chains keep per-slot state *on the device* — that audit is a separate scope (below) |
| `m->seen[]`, `momentum_logits`, `m->first_step` | no (process-global) | reset per-request inside `q36_serve_start` in both modes already; unchanged |

**Conclusion: the gate is conservatism, not a correctness wall.** The disk
cache reads and writes exactly the per-slot state that `q36_seq_swap` already
trades, and prefills never interleave in mux. The failure paths are the same
ones single-slot already relies on: a restore whose payload hash fails has
dirtied only the bound slot's K/V + DeltaNet — buffers the `!reuse` branch
(`reset_recurrent`, `kv_len=0`, full prefill) rewrites from scratch, and that
slot's live state was already unusable (the cache probe runs only after live
reuse *and* the pin both missed).

## Risks and how the design contains them

1. **A corrupted snapshot dirties a slot, and slots are a shared resource.**
   Single-slot has the same exposure with one state; in mux the dirty slot is
   one of N. Mitigation: the existing recovery (full prefill from the reset)
   makes the slot usable again — no cross-slot contamination is possible
   because restore never writes outside the bound slot. Validation must prove
   the recovery, not just the happy path (truncate a snapshot file mid-payload).
2. **The Vulkan/CUDA chains hold slot state on the device.** `q36_pin_state_save`
   pulls the device's newest bytes on the single-slot device paths, but the mux
   chains (`q36c_mux_*`, `q36cc_mux_*`) keep each conversation resident on the
   device — a restore would have to invalidate/refresh the right slot's device
   rows. **Proposed scope: CPU chain only** (this host serves on the CPU);
   keep the device chains under the `KV_SLOTS==1` gate exactly as today. The
   condition is already expressible in-engine (the chain globals).
3. **LRU/byte budget across slots.** Pruning runs synchronously inside a
   prefill (single-threaded engine), so no interleaving hazard. Files are
   prompt-keyed, not slot-keyed — two slots hitting the same prompt share one
   snapshot, which is correct and desirable. No change.
4. **Output-identity contract.** A restore claim needs output-identity
   evidence per the house contract; nothing about mux changes what a correct
   restore is — the snapshot must reproduce the same continuation as the
   engine's own recomputation.

## Proposed change (small)

Lift the two `g_q36_mux_slots == 1 &&` conditions in `q36_serve_start` to a
shared helper — roughly: disk cache allowed when serving, on any slot count,
on the CPU chain (no Vulkan/CUDA chain active), with every other existing gate
unchanged (no vision payload, no sliding-window, no pin, no logprobs, not
tainted, min-token floor, cache on). Estimated ~10–20 lines in `c/qwen36.c`,
no gateway changes, no file-format changes (the `Q36PC01` header/geometry key
already guards against cross-geometry accidents).

## Validation protocol (per the house rules — every claim needs output-identity evidence)

Driver: a mux-aware extension of the existing cold/restore matrix
(`c13_matrix.py` pattern), two fresh restarts per arm, **interleaved A/B**
(`KV_SLOTS=2`, `COLI_PROMPT_CACHE=0` vs cache on with the default budget):

1. **Cross-restart restore in mux (the headline):** conversation A and B run
   turns in mux; the process restarts; both conversations' exact repeats must
   restore from disk in mux (`reuse_source=disk_snapshot`) with
   output-identical text vs their own cold runs.
2. **Live-reuse non-regression:** within-process consecutive turns in mux must
   still hit `pin`/live reuse, *not* the disk path (reuse source census).
3. **No cross-contamination:** two different conversations on two slots —
   each turn's output sha must equal its own cold reference; neither
   conversation may ever restore a snapshot of the other's prompt (the ids
   probe forbids it by construction; assert via the reuse census).
4. **Corruption recovery:** truncate one snapshot file's payload, re-request —
   the request must fall through to a full prefill and produce the cold
   output exactly; the slot must serve a following conversation normally.
5. **Mux decode regression:** concurrent two-slot decode rates within the
   recorded scatter of the cache-off arm (decode claims need distributions,
   per the C14 finding).

Effort estimate: implementation ~half a day including the fixture-suite run;
the validation matrix ~1.5–2 h of engine time at 4K/12K-class prompts
(12K prefills on the cold arm are the long pole; the restore arms are fast).

## Recommendation

Proceed, CPU chain only, with the five-point validation above. The change is
contained, the hazard analysis says the existing single-slot recovery argument
transfers to slots unchanged, and the win is the one the real Copilot session
already showed mattering (process restarts threw away both conversations; the
disk cache is what survives them). The device-chain audit (Vulkan/CUDA mux) is
explicitly out of scope and stays gated.
