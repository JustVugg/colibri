# E5 — MTLResidencySet over the existing malloc'd slabs (experiment branch)

Branch: `e5/metal-residency-set` (cut from `origin/dev` @ `caa49f7`, per spec — E4 was cut
from `main` @ `72d3d37`; `backend_metal.mm`/`.h` are byte-identical between the two bases,
confirmed via `git diff 72d3d37 origin/dev -- c/backend_metal.mm c/backend_metal.h`).

## The hypothesis

E4 (MTLHeap-backed slabs) proved that batching residency declaration kills the GPU stall
(25.9s → 3.9s at cap16, −85%), but changing the *allocation* (heap sub-buffers instead of
malloc'd host memory) brought a +12–13s expert-disk-load tax, suspected first-touch/lock
contention on CPU-writes into GPU-owned heap pages. E5 decouples the two: keep the exact
same malloc'd slabs and per-slab `newBufferWithBytesNoCopy`-wrapped `MTLBuffer`s, and change
**only** residency bookkeeping — declare residency once, ahead of time, on a set attached to
the command queue, instead of once per command buffer via `useResource:`. If the stall
reduction survives without the load-path tax (malloc pages never change ownership), E5 wins.

## What changed

All mechanism code is confined to `c/backend_metal.mm` —
`coli_metal_register`/`coli_metal_unregister`'s existing signatures and every call site in
`colibri.c` (expert_load, uring_load_add, qalloc, kv_alloc, map_of_fd) are untouched; the
residency-set bookkeeping lives entirely inside those two functions' existing bodies. The
`colibri.c`/`backend_metal.h` touches are two, both coordinator-sanctioned: the validator
round-1 instrumentation hook (`coli_metal_resset_stats` + the gate-on-only `METAL-RESSET:`
stats line in `profile_print`) and the ported fslab-OOM unwind fix (see "Validator round 1
fixes" item 4). Still a smaller diff shape than E4,
which needed a new alloc/free API and four new `glm.c` call-site arms because it changed the
allocation function itself.

Env-gated `COLI_METAL_RESSET=1`, default OFF, runtime `@available(macOS 15.0, *)` guard with
a one-line stderr fallback when requested on an older OS or when residency-set creation
fails. Gate off ⇒ every new branch is skipped and behavior is byte-for-byte the stock path
(verified by inspection: `g_resset_enabled` starts `false` and nothing sets it except inside
the `COLI_METAL_RESSET` `getenv` branch in `coli_metal_init`, so `resset_add`/`resset_remove`/
`resset_flush` are no-ops and `moe_submit`'s `useResource:` loop runs unconditionally).

### Lifecycle (`c/backend_metal.mm`)

- **Init** (`coli_metal_init`, end of the existing pipeline-setup `@autoreleasepool`): if
  `COLI_METAL_RESSET=1` and `@available(macOS 15.0, *)`, create one
  `MTLResidencySetDescriptor` (`initialCapacity=4096`, a presize hint only), call
  `[g_dev newResidencySetWithDescriptor:desc error:&err]`, and `[g_queue addResidencySet:rs]`
  — one set, attached once, for the process lifetime. Failure (old OS or creation error)
  prints one stderr line and leaves `g_resset_enabled=false` — stock path.
- **`coli_metal_register`**: after wrapping the buffer exactly as today
  (`newBufferWithBytesNoCopy`) and pushing the `g_slabs` entry under `g_slab_mtx` exactly as
  today, calls `resset_add(b)` **after dropping `g_slab_mtx`** but before returning.
  `resset_add` takes a dedicated `g_resset_mtx` (guarding only the set mutations and the
  dirty flag), calls `[rs addAllocation:b]` and sets `g_resset_dirty` — **it does not
  commit**. No Metal call ever runs under `g_slab_mtx` (validator round-1 fix; E4's audit
  round 2 identified mutex-over-live-Metal-call as the leading suspect for its +12s
  expert-disk regression). Re-registering a live base (no in-tree caller does today) drops
  the replaced wrapper from the set via `resset_remove(old)` before adding the new one
  (hazard-audit defensive fix — the set would otherwise retain the old buffer, and its
  pages' residency, forever), keeping set membership an exact mirror of `g_slabs`.
- **`coli_metal_unregister`**: erases the `g_slabs` entry under `g_slab_mtx` (stashing the
  buffer), then calls `resset_remove(b)` **outside `g_slab_mtx`**, before returning.
  `resset_remove` (under `g_resset_mtx`) calls `[rs removeAllocation:b]` **and commits
  immediately** — no batching — because the caller frees the host memory right after the
  function returns. See UNCERTAINTIES for why this asymmetry is deliberate.
- **`moe_submit`** (the one function whose `use` list — resolved expert weight/scale slabs —
  scales with LRU cache size): calls `resset_flush()` at the top (commits any pending adds
  from `resset_add`, under `g_resset_mtx` — it never touches the slab lock), then, if
  `g_resset_enabled`, **skips** the
  `for(auto&b:use) [e useResource:b usage:MTLResourceUsageRead];` loop entirely — residency
  is already guaranteed by the queue-attached set. Every other `useResource:` call site in the
  file (`bind_gemv`'s weight/scale buffers, `coli_metal_attn_decode`/`coli_metal_layer_decode`'s
  `Lb`/`Rb`/`kvbW`/`kvbS`/`inB`/`pnB`/`rwB`/`rbB`, `coli_metal_gemm`'s `wb`/`sb`) is
  **left completely unchanged**, regardless of the flag — see "Why only `moe_submit`" below.
- **Shutdown** (`coli_metal_shutdown`): `[g_queue removeResidencySet:rs]` then clears the
  globals, ahead of the existing `g_queue=nil; g_dev=nil;`.

### Why only `moe_submit` skips `useResource:`

Apple's `MTLResidencySet` class reference (developer.apple.com, fetched during design on
2026-07-18) is explicit: *"Residency sets don't support hazard tracking, so you need to
account for hazards with fences and events."* The SDK header on this box
(`MTLResidencySet.h`, read directly) is **silent** on hazard tracking — the statement comes
from Apple's online documentation and adoption guide ("Simplifying GPU resource management
with residency sets"), not the header (see UNCERTAINTIES for sourcing). Dropping
`useResource:` therefore risks losing whatever hazard-tracking value those calls provided. Rather than apply the residency set uniformly and
argue *in general* that hazard tracking isn't load-bearing, this diff draws the line at the
one call site the mechanism history actually implicates:

`moe_submit`'s `use` vector holds only **read-only** (`MTLResourceUsageRead`), **indirectly
referenced** slab buffers — the kernel (`moe_gemv`) never touches them via `setBuffer:`; it
dereferences raw GPU addresses (`waddr[e]`/`saddr[e]`) baked into a separately-bound address
array (`bag`/`bau`/`bad`/`bsg`/`bsu`/`bsd`), which is exactly the "indirect reference" case
`useResource:` exists for. No GPU-side write ever touches these buffers, so there is no
write-after-write/read-after-write hazard for Metal's tracking to have been serializing in
the first place; the one real hazard — a slab unregistered+freed+reused by the CPU while an
async in-flight `moe_block_begin` command buffer still references it via a baked-in GPU
address — is a **CPU-write race that Metal's hazard tracking never protected against anyway**
(hazard tracking only covers GPU-side command dependencies visible through the Metal API; a
raw host-memory write via `pread`/`memcpy` is invisible to it regardless of `useResource:`).
That race is, and always was, the engine's own responsibility (slot/generation lifecycle: a
slab isn't freed while an outstanding async handle still owns it) — unrelated to E5.

Every other call site (`bind_gemv`, attention K/V cache writes) either doesn't scale with
cache size (fixed per-layer dense tensors — no perf benefit to touching) or has real
GPU-side write traffic in the same encoder (`Lb`/`Rb` are written by `a_copy` and read by
`a_score`/`a_clat` within one encoder — currently ordered by explicit
`memoryBarrierWithScope:MTLBarrierScopeBuffers` calls already present in `encode_attention`,
not by `useResource:`'s hazard tracking, but touching them wasn't needed for the hypothesis
and was judged not worth the added surface area). Leaving them untouched keeps the diff's
blast radius matched to the one seam the fix-plan's v5 finding actually names.

### Deferred-commit design (`resset_add` batches; `resset_remove` doesn't)

`coli_metal_register` is called from parallel OpenMP loader threads in tight bursts
("warmup fan-out" — same phrase E4's audit used for the same threads). Committing on every
single `addAllocation:` would reintroduce a per-slab cost on the load path, which is exactly
what E4's own +12s regression looked like (mutex held across a live Metal call, serializing
loader threads). So `resset_add` only marks `g_resset_dirty`; the commit is deferred to the
next `moe_submit` call, which flushes once via `resset_flush()` before it relies on the set
for residency.

This is correct — not just fast — because of an existing invariant the codebase already
depends on for `resolve()` to work at all: a slab's `coli_metal_register` call always
completes — including its trailing `resset_add`, which runs after `g_slab_mtx` is dropped
but **before the function returns** — before any dispatch that references that slab's
pointer can call `resolve()` for it (the caller in `colibri.c` cannot pass a freshly-loaded
expert's pointer to a dispatch before the load — which registers it — returns). After the
validator round-1 mutex split, the flush's synchronization runs through `g_resset_mtx`
alone: `resset_add`'s set mutation + dirty write and `resset_flush`'s dirty read + commit
are serialized by that one mutex, whose release/acquire pairs provide the memory ordering;
`g_slab_mtx` still orders the slab-table bookkeeping (register-before-resolve) exactly as on
stock. So any slab a given `moe_submit` invocation will resolve was `addAllocation:`-ed (and
marked dirty) strictly before that invocation's `resset_flush()` acquired `g_resset_mtx` —
the flush is guaranteed to cover it, regardless of what other threads are concurrently
registering unrelated slabs. The two mutexes are never held simultaneously anywhere, so no
deadlock ordering exists to maintain.

`resset_remove`, by contrast, commits synchronously and immediately, with no batching,
because the caller (`colibri.c`, in every one of the four slab-realloc call sites, and in
`kv_alloc`) frees the underlying host memory *right after* `coli_metal_unregister` returns.
An uncommitted-but-still-set-member allocation pointing at memory the host has already freed
is a potential use-after-free the GPU could act on — deferring that removal is not a
performance-vs-safety tradeoff, it's just unsafe, so it isn't deferred. (The spec's own
lifecycle wording backs this reading: "`coli_metal_register` → add allocation + commit
**(batch commits where call pattern allows)**" carries a batching allowance that
"`coli_metal_unregister` → remove + commit" does not.)

## Instrumentation parity

No existing counter's semantics changed. `coli_metal_moe_times`/`coli_metal_moe_counts`
(`g_t_setup`, `g_t_gpu`, `g_t_kernel`, `g_t_scatter`, `g_moe_ok`/`g_moe_fb`/`g_moe_experts`)
are computed exactly as before — `resset_flush()` runs *before* `ts_start = mnow()` in
`moe_submit`, so its cost is **outside** `g_t_setup`, keeping the orchestrator's A/B harness
reading the same counters with the same meaning across stock/E4/E5. The flush cost is
surfaced separately (validator round-1 fix — the original design left it invisible, a blind
spot for the battery): a dedicated `g_t_resset_flush` accumulator timed around the flush in
`moe_submit`, exported via `coli_metal_resset_stats()` (`backend_metal.h`) and printed by
`profile_print` as its own `METAL-RESSET: flush N.NNs` line — a **separate line following
the `METAL:` line, mirroring E4's `METAL-HEAP:` convention, so the existing `METAL:` line
the harness parses keeps its exact format** — printed **only when the gate is on** (the
function returns 0 when off), so stock output stays byte-identical. The register-side
`resset_add`/`resset_remove` costs have no dedicated counter: they run inside the engine's
existing expert-load wait accounting (the `t_ewait` window in `colibri.c`), noted in a comment
at `resset_add`, so a load-path regression from set bookkeeping would already show in the
existing disk/wait numbers. `[METAL] residency-set: on` / the two fallback stderr lines from
`coli_metal_init` confirm which path a run took.

## Validator round 1 fixes

1. **REQUIRED, Metal calls hoisted out of `g_slab_mtx`** (`backend_metal.mm`): the original
   design ran `addAllocation:`/`removeAllocation:`/`commit` while holding `g_slab_mtx`, the
   lock the parallel OMP loader threads contend on — structurally identical to the
   mutex-over-live-Metal-call shape E4's audit round 2 identified as the leading suspect for
   its replicated +12s expert-disk regression, and the SDK header notes commit on a resident
   set tries to make resources resident "instantly" (real synchronous work; this set is
   resident from startup since it is queue-attached for the process lifetime). Fixed by
   introducing a dedicated `g_resset_mtx` guarding only the set mutations + dirty flag;
   `g_slabs` push/erase stays under `g_slab_mtx` exactly as stock; the two mutexes are never
   held together. The register→flush→resolve happens-before argument is preserved — see the
   updated "Deferred-commit design" section and the comment at `resset_add`.
2. **REQUIRED, false citations corrected** (this file + the `moe_submit` commit message,
   rewritten pre-push): the original text attributed the hazard-tracking and thread-safety
   statements to the SDK header (`MTLResidencySet.h`), which is in fact silent on both
   topics. The statements come from Apple's **online** `MTLResidencySet` class reference and
   the "Simplifying GPU resource management with residency sets" adoption guide (both
   fetched 2026-07-18 during design). All attributions now name the actual source; where a
   claim rests on design reasoning rather than documentation, it is labeled as such.
3. **REQUIRED, flush cost made harness-visible**: `g_t_resset_flush` +
   `coli_metal_resset_stats()` + the gate-on-only `METAL-RESSET:` line in `profile_print` —
   see "Instrumentation parity" above.
4. **Pre-existing fslab OOM-unwind bug — now CARRIED ON THIS BRANCH** (follow-up commit,
   coordinator-sanctioned second `colibri.c` change): `expert_load`'s fslab OOM path
   (`c/colibri.c`, in `expert_load_impl`) freed `s->slab` via `compat_aligned_free` **without**
   `coli_metal_unregister` — on stock that leaves a stale `g_slabs` entry whose GPU
   exposure ends with the last command buffer that declared it; under E5 the buffer would
   additionally be a **permanent residency-set member** referencing freed host memory until
   some later realloc of the same slot unregisters by pointer, a strictly longer-lived
   exposure than stock's transient per-CB one. Fixed by porting E4's reference
   implementation (`6753225`) to dev's non-heap code shape: `coli_metal_unregister(s->slab)`
   before the free. The `uring_load_add` analog (E4's audit round-2 "cheap insurance") is
   deliberately NOT carried: that arm is `#ifdef __linux__`-gated and `COLI_METAL` is
   macOS-only, so it is dead code on every real build target, and unlike E4 this branch has
   no allocation-path reason to touch the function at all.

## Per-seam differences vs E4

| Seam | E4 (`e4/metal-heap`) | E5 (this branch) |
|---|---|---|
| Allocation | New: `MTLHeap` sub-buffers via `coli_metal_heap_alloc` | Unchanged: same `posix_memalign` + `newBufferWithBytesNoCopy` |
| Coordinator C source / `backend_metal.h` | `glm.c` touched (new alloc/free API, 4 call sites + `expert_host_release`) | `colibri.c` + header touched only for instrumentation and the OOM-unwind fix |
| Residency scope | Declared once **per command buffer** (`useHeap:`, still inside `moe_submit`) | Declared once **for the process** (queue-attached set), refreshed incrementally at register/unregister |
| Hazard tracking | Heap sub-buffers forced `MTLHazardTrackingModeUntracked` always (allocation-level) | Untouched at the resource level; `moe_submit` alone stops calling `useResource:` (encoder-level), independent of `COLI_METAL_UNTRACKED` |
| Per-buffer vs per-set skip | `[b heap]` (Metal's own `MTLResource.heap` property) checked per buffer — heterogeneous mixes possible if a slab fell back to malloc | Blanket `if (!g_resset_enabled)` — homogeneous by construction, since every registered slab goes through the same `coli_metal_register` path when the gate is on |
| Availability guard | None needed (`MTLHeap` is old API) | `@available(macOS 15.0, *)`, matching this box's macOS 26.5 but required for portability |
| Known regression | +12–13s expert-disk load at cap16 (suspected first-touch/lock contention on heap pages) | None expected — malloc pages never change ownership; **unverified without a run** |

## What to measure (orchestrator, cap1/cap16, stock vs E4 vs E5)

1. **GPU stall** (`coli_metal_moe_times` gpu/kernel breakdown) — success: E5 ≈ E4's
   −85%-class reduction vs stock at cap16.
2. **Expert-disk load path** (existing load/service-time counters) — success: E5 ≈ stock,
   i.e. **no** repeat of E4's +12–13s tax, since allocation is untouched.
3. **tok/s** — should track (1) and (2) together.
4. **md5 within a fixed dispatch composition** — flag on vs off must be byte-identical at a
   given cap (the "Output-invariant by construction" hard constraint); flag-on vs flag-on
   across cap1/cap16 may legitimately differ (different dispatch composition, per the
   fix-plan's "Determinism side-finding").
5. **`[METAL] residency-set: on` line present in stderr** at flag-on startup, and absent
   (or the OS<15/create-failed fallback line) otherwise — cheap sanity check that a run
   actually exercised the intended path before trusting its numbers. Also read the
   **`METAL-RESSET: flush` line** (gate-on only): if that number is large, the deferred
   set-commit cost is eating the stall win from the dispatch side.
6. If the hypothesis holds (E5 stall ≈ E4, E5 load-path ≈ stock, identical output), E5 becomes
   the upstream PR candidate and must include the cap-default recalibration flagged in PR
   #386's CURRENT-STATE CALIBRATION markers, per the spec's validation plan.

## Build

`cd c && make glm METAL=1` and a separate explicit `-Wall -Wextra` compile of
`backend_metal.mm` (the Makefile's `METALXX` line does not itself pass `-Wall -Wextra`, so
the warning surface was checked with those flags added explicitly; current `dev` contributes
one pre-existing `unused variable 'TG'` warning), plus
`cd c && make glm` (plain, non-Metal — the one `colibri.c` instrumentation touch, the `METAL-RESSET` stats line,
is inside the pre-existing `#ifdef COLI_METAL` arm of `profile_print`, so the plain build
compiles none of it), and
`make metal-test` (existing synthetic kernel-correctness unit test — no model, no
`glm52_i4/`, random weights — run once with `COLI_METAL_RESSET` unset and once with
`COLI_METAL_RESSET=1` to numerically exercise `coli_metal_register`/`moe_submit`'s changed
code path, since the task scope excludes running the real model). Exact results in the final
report, not here (build results belong to the report per the task's deliverable split, and
this file is written before the batched build run, per the scheduling constraint).

## UNCERTAINTIES

**Everything below is a judgment call, a seam where the residency-set lifecycle interacts
with the existing queue/command-buffer structure, or something unverifiable without a real
model run — flagged per the task's hard requirement.**

1. **The central design risk: skipping `useResource:` in `moe_submit` gives up Metal's
   automatic hazard tracking for that buffer set.** Sourcing (corrected in validator round
   1): the SDK header on this box
   (`/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/.../Headers/MTLResidencySet.h`,
   read directly) documents the protocol only in terms of residency and says nothing about
   hazard tracking either way; the two operative statements are from Apple's **online**
   documentation (fetched 2026-07-18): the "Simplifying GPU resource management with
   residency sets" adoption guide — *"You don't need to call `useResource`/`useHeap`... for
   allocations in a residency set"* — and the `MTLResidencySet` class reference —
   *"Residency sets don't support hazard tracking, so you need to account for hazards with
   fences and events."* I reasoned through every
   code path that touches `moe_submit`'s `use` buffers (read-only, indirectly referenced,
   never concurrently written, freed only after the engine's own slot lifecycle guarantees
   no outstanding async reference) and concluded removing `useResource:` there specifically
   is safe — but this reasoning is **not the same as having run the model**. If any code
   path I didn't trace lets a slab get unregistered while an async `moe_block_begin` handle
   is still in flight and reading it, this change removes a mitigation (weak as it may have
   been) that existed before. **This is the #1 thing to watch for md5 divergence on**, and
   the reason the scope was deliberately narrowed to `moe_submit` alone rather than applied
   uniformly.
2. **Residency-set mutations are serialized under a dedicated `g_resset_mtx` (validator
   round-1 fix — originally they ran under `g_slab_mtx`, the E4-regression shape; no Metal
   call runs under the slab lock anymore).** The serialization itself is kept as required
   for correctness: Apple's online `MTLResidencySet` class reference states the set's
   *"methods aren't thread-safe"* (the SDK header contains no thread-safety statement either
   way — citation corrected in round 1; the online doc is the source). What remains
   **unverified without profiling a loaded run** is the *cost* of the calls themselves:
   `resset_remove`'s synchronous `commit` runs inside `coli_metal_unregister` on the
   loader path (its cost lands in the existing `t_ewait` accounting), and the SDK header
   says commit on a resident set tries to make added/removed resources resident/non-resident
   *"instantly"* — real synchronous work, since this set is resident from startup
   (queue-attached for the process lifetime). If `commit()`/`addAllocation:` turn out
   expensive on this hardware/OS build, the load path degrades through set bookkeeping
   rather than mutex contention — a different, now-decoupled failure mode, but the same
   symptom as E4's regression. Orchestrator: check E5's load-path timing against stock, not
   just against E4, and read the new `METAL-RESSET: flush` line for the dispatch-side share.
3. **`resset_flush()`'s cost sits outside `g_t_setup`/the `moe_times` breakdown** (it runs
   before `ts_start = mnow()`), by design, to keep the harness's existing counters
   meaningful — and, since validator round 1, it is **no longer invisible**: the
   `g_t_resset_flush` accumulator surfaces it as the gate-on-only `METAL-RESSET: flush`
   line (see "Instrumentation parity"). Residual blind spots: (a) the accumulator is a
   plain double written from `moe_submit` on the engine thread, matching the existing
   `g_t_setup` convention — if `moe_submit` were ever called from multiple threads
   concurrently, both counters would be equally wrong; (b) the register-side
   `resset_add`/`resset_remove` costs have no dedicated counter and are only visible
   blended into the existing `t_ewait`/disk-wait numbers (comment at `resset_add` says so)
   — a fine-grained attribution would need a throwaway probe.
4. **`initialCapacity = 4096` on the `MTLResidencySetDescriptor` is an unverified guess.**
   It's documented as a presize hint only (no correctness effect either way), chosen to be
   "clearly larger than the permanent-weight-tensor + KV-cache + plausible cap16 LRU-slab
   count" without actually counting those registrations precisely. Too small just means
   internal array growth; not a correctness concern, flagged only because it's a number I
   picked without measuring.
5. **Not calling `requestResidency()` proactively.** Apple's guide frames it as an optional
   latency-hiding call ("call ahead of time during non-critical moments... to minimize [first
   command buffer] latency"), and Blender's Cycles PR (the spec's cited reference
   implementation) doesn't appear to use it either per its PR description. Omitted to keep
   the lifecycle minimal and match the reference pattern; if profiling shows a
   first-command-buffer-after-a-load-burst latency spike, this is the documented lever to try
   next, not implemented here.
6. **The deferred-commit correctness argument (item in "Deferred-commit design" above) rests
   on a single-writer-before-single-reader program-order guarantee that is true today by
   inspection but is not an invariant enforced anywhere in code** (no assertion, no type-level
   guarantee) — it's the same kind of implicit ordering `resolve()` itself already depends on
   for correctness (a slab must be registered before any dispatch can resolve its pointer),
   so this diff doesn't introduce a new category of fragility, but it's worth naming
   explicitly rather than leaving implicit.
7. **Async `moe_block_begin`/`moe_block_end` overlap with concurrent `register()` calls**
   (background loader threads registering new/different experts while an unrelated MoE block
   is still in flight on the GPU) was reasoned through but never exercised in a real
   concurrent stress scenario — the synthetic `metal-test` unit test's `run_moe` calls are
   single-threaded and synchronous (`coli_metal_moe_block`, not the async `_begin`/`_end`
   pair), so it does **not** cover this interleaving. The real engine's `PILOT`/prefetch and
   `moe_block_begin`/`_end` overlap path is exactly the concurrency shape most likely to
   expose a bug in this design if one exists, and is untested here by construction (out of
   scope: no model runs).
8. **`coli_metal_gemm` (prefill path) and `bind_gemv` (attention path) still call
   `useResource:` unconditionally, so they get no CPU-overhead benefit from the residency set
   even though their buffers are also set members.** This is deliberate (see "Why only
   `moe_submit` skips" above) but means E5's win, if any, is scoped to the decode-path MoE
   dispatch loop specifically — prefill and attention timing should be unaffected by the flag,
   which is itself a testable prediction the orchestrator's harness can check.
9. **API surface verified against this box's actual SDK headers**
   (`MTLResidencySet.h`, `MTLDevice.h`, `MTLCommandQueue.h`, `MTLAllocation.h`,
   `MTLResource.h` — all read directly, not from memory) and against Apple's own
   "Simplifying GPU resource management with residency sets" guide, so the method names/
   signatures (`newResidencySetWithDescriptor:error:`, `addResidencySet:`,
   `removeResidencySet:`, `addAllocation:`, `removeAllocation:`, `commit`) are
   high-confidence. What is **not** independently verified is runtime behavior beyond what
   the docs state and what the synthetic unit test exercises — no substitute for the
   orchestrator's real cap-sweep battery.
10. **Pre-existing fslab OOM-unwind bug — carried on this branch** (follow-up commit; see
   "Validator round 1 fixes" item 4 for the full mechanism). The one-line
   unregister-before-free fix from E4's `6753225` is ported to dev's non-heap code shape,
   so the upstream PR built from E5 inherits it automatically. Residual notes: (a) the fix
   is only reachable through the fslab-OOM path (allocation failure mid-load), so it is
   untestable without an OOM-injection harness and cannot affect the orchestrator's
   controlled A/B runs at sane RAM headroom — carried as correctness insurance, verified by
   inspection + clean builds only; (b) the `__linux__`-gated `uring_load_add` analog is
   deliberately not carried (dead code on every real build target — rationale in the fixes
   section).
