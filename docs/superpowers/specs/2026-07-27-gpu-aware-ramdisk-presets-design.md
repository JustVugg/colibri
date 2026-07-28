# GPU-aware RAM-disk Presets

Date: 2026-07-27

## Problem

The RAM-workspace TUI currently opens a six-step editor with low-level placement,
capacity, and runtime defaults. An operator must understand Colibri's shared
versus replicated topology, NUMA placement, memory reserves, profile-guided
partial staging, and managed runtime settings before reaching an exact plan.

The intended deployment has four CPU/NUMA domains and two GPUs. One managed
Colibri engine uses both GPUs, and the primary optimization target is transfer
from RAM-staged tensor data to VRAM. The current `per-node` topology does not
represent that workload: it creates a complete model replica and an independent
engine per selected NUMA node.

The TUI should ask one intent-level question before the normal workflow,
prepopulate the authoritative settings, and proceed directly to exact plan
review. It must not imply that generic replication or multiple tmpfs mounts
automatically make a single multi-GPU engine faster.

## Goals

- Replace the initial low-level setup walk with one first-run preset choice.
- Make **Fastest GPU staging** the default choice.
- Detect GPU-to-NUMA locality and use only GPU-local NUMA nodes for automatic
  shared placement.
- Keep one shared model copy and one multi-GPU engine in the automatic mode.
- Enable the managed CUDA upload path instead of the CPU-only direct RAM-map
  path when Fastest GPU staging is selected.
- Fall back from full to profile-guided partial staging when a full copy cannot
  fit safely.
- Keep replication an explicit advanced selection that Auto never chooses.
- Preserve exact plan review, editable advanced settings, planner blockers, and
  the existing preparation confirmation.
- Give the Textual and curses frontends identical preset semantics.

## Non-goals

- Do not create RAID-like striping across multiple tmpfs mounts. tmpfs mounts
  are not independent storage devices, and extra mounts do not add bandwidth
  by themselves.
- Do not add per-GPU replicas to one engine or teach the tensor loader to select
  a different source replica for each destination GPU.
- Do not migrate individual staged tensor pages to a destination GPU's NUMA
  node in this change.
- Do not benchmark alternative NUMA layouts automatically before preparation.
- Do not change the scriptable lifecycle commands or add a public `--preset`
  CLI contract.
- Do not bypass memory admission, NUMA namespace validation, exact review, or
  the destructive-action confirmation.

## User experience

### First-run question

When no RAM-workspace manifest exists, both terminal frontends first ask:

```text
What should Colibri optimize?
```

The choices are:

1. **Fastest GPU staging (default)** — one GPU-aware shared copy and one
   multi-GPU engine.
2. **Single RAM copy** — the existing full shared placement.
3. **Minimal RAM** — the largest safely admitted profile-guided partial
   placement.
4. **Multiple NUMA replicas (advanced)** — the existing complete-copy and
   independent-engine-per-node topology.

The choice only populates a draft. It performs no mount, copy, engine launch, or
other privileged action.

After selection, the TUI builds the authoritative plan and jumps directly to
the exact review step. All existing advanced placement, capacity, and runtime
fields remain available. Editing a preset-controlled field marks the draft as
**Custom** without resetting the operator's other values.

If a manifest already exists, the TUI skips the question and opens the
persisted lifecycle/status view. A preset must never rewrite a prepared,
running, stopped, or error manifest.

### Review evidence

For a GPU-aware draft, review shows:

- detected GPU indices and names;
- each GPU's PCI bus ID and resolved Linux NUMA node;
- selected GPU-local memory nodes;
- selected whole-core CPU mask;
- one shared model copy and one managed engine;
- full or partial staged bytes and remaining headroom;
- CUDA enablement, automatic VRAM budget, and mmap upload mode;
- the reason for any fallback; and
- a warning when GPU locality or CUDA capability could not be proven.

The existing plan token and preparation confirmation remain mandatory.

## Preset resolution

Preset resolution is a shared, pure planning layer used by both frontends. It
receives discovered hardware, analyzed model facts, the current draft
arguments, and the optional usage profile. It returns populated arguments,
preset metadata, and explanations, then calls the existing authoritative
planner. Frontends do not duplicate memory formulas or decide whether a plan
is safe.

### Fastest GPU staging

GPU discovery extends the existing NVIDIA query with PCI bus IDs. On Linux,
each PCI device is resolved through:

```text
/sys/bus/pci/devices/<domain:bus:device.function>/numa_node
```

Duplicate GPU-local nodes are collapsed while preserving deterministic device
order. A reported node of `-1` maps to the only effective NUMA node on a
single-node host. A GPU node outside the process's effective memory-node mask
is not silently used.

When GPU locality and a CUDA-capable engine are available, the preset sets:

- topology to shared/interleaved;
- memory nodes to the unique effective nodes local to the selected GPUs;
- CPU affinity to complete physical cores belonging to those nodes;
- one managed engine using every selected GPU;
- managed CUDA enablement;
- automatic VRAM expert-tier sizing;
- asynchronous CUDA transfers;
- `COLI_MMAP=1` for staged-file upload; and
- `COLI_RAMMAP=0`, because the direct RAM-map expert tier is a CPU-resident
  path and currently excludes those experts from the VRAM hot tier.

For the target four-node/two-GPU host, this produces one model copy interleaved
only across the two GPU-attached NUMA nodes. The two CPU-only nodes are excluded
from staged source placement.

This arrangement minimizes distance with the current single-source engine, but
does not claim that every page is local to both GPUs. A tensor interleaved
across two nodes can still contain remote pages for either GPU. Exact
per-destination locality requires a later engine-level tensor-placement
feature.

The resolver first asks the existing planner for a full-copy candidate. It
accepts that candidate only when it has no blockers. If full staging is unsafe,
the resolver asks for the largest safe profile-guided partial candidate.
Capacity selection uses the authoritative planner iteratively rather than
duplicating reserve calculations, and respects the planner's shard-closure
granularity.

If no positive partial candidate fits, or no compatible usage profile exists,
the review is blocked with an actionable explanation. The preset never
silently switches to replication.

If GPU discovery, PCI locality, or CUDA capability is unavailable, Auto falls
back to the Single RAM copy draft and records a prominent explanation. It does
not claim GPU-optimized staging. On a multi-node host with an unresolved GPU
node, the fallback uses the existing effective-node shared policy and warns
that GPU locality is unknown.

### Single RAM copy

This preset reproduces the existing conservative draft:

- shared/interleaved topology;
- full staging;
- one model copy and one engine;
- effective memory nodes and whole physical cores; and
- existing safe defaults for prefaulting, huge pages, copy workers, context,
  and ports.

It remains available as an explicit non-GPU-aware choice.

### Minimal RAM

This preset uses:

- shared/interleaved topology;
- profile-guided partial staging;
- the existing effective memory nodes and whole physical cores; and
- the largest positive capacity admitted by the planner after runtime and
  operating-system reserves.

The staged set remains a deterministic shard closure derived from the compatible
usage profile. If no compatible profile exists, review is blocked and explains
how to generate one. Minimal never falls back to a full copy because that would
violate the operator's stated memory objective.

### Multiple NUMA replicas

This explicit advanced preset selects the existing `per-node` contract:

- one complete staged copy per selected NUMA node;
- one independent managed engine per selected node;
- node-local CPUs and strict memory binding; and
- multiplied RAM, mount, port, and endpoint counts.

Auto never selects this preset. Review labels it as replication, not sharding
or single-engine GPU locality, and retains the current multiplied-memory danger
state and confirmation.

## Managed runtime contract

The authoritative plan persists a managed accelerator section containing the
selected mode, devices, GPU-to-NUMA evidence, mmap/RAM-map choice, asynchronous
copy setting, and VRAM-budget policy. Lifecycle actions consume only this
persisted section; they do not inherit ambient accelerator variables.

For Fastest GPU staging, managed start and benchmark explicitly apply the
reviewed CUDA settings. The launch sanitizer continues to remove ambient
`COLI_CUDA`, `COLI_GPU(S)`, `CUDA_EXPERT_GB`, `COLI_MMAP`, and related values
before applying the persisted managed values. This preserves reproducibility
while fixing the current behavior in which managed launch removes GPU settings
without replacing them.

Other presets retain their existing managed runtime behavior unless their
advanced draft explicitly contains an accelerator plan.

## Failure behavior

- Preset resolution itself is read-only and cannot leave a partial workspace.
- Invalid or stale form edits continue to disable lifecycle actions.
- An unavailable model, invalid NUMA mask, incomplete physical-core selection,
  insufficient memory, missing profile, or unusable CUDA plan appears as a
  planner blocker before preparation.
- A GPU disappearing or moving outside the effective device/NUMA namespace
  between review and start fails closed during lifecycle revalidation.
- Preparation and startup retain their existing rollback and error-manifest
  behavior.

## Alternatives considered

### Frontend-only field templates

Each frontend could directly assign topology, mode, and capacity fields. This
is smaller, but duplicates policy and cannot guarantee that managed launch
uses the reviewed GPUs. It was rejected because the two TUIs would drift and
the Fastest GPU staging label would be misleading.

### Automatically select per-node replicas

Replicas can improve local access for independent node-local engines, but the
target workload is one engine using both GPUs. Automatic replication would
multiply RAM and endpoints without solving that contract. It was rejected and
remains an explicit advanced choice.

### Multiple striped tmpfs mounts

Extra tmpfs mounts do not provide independent queues or controllers. Equal
page interleave already uses multiple selected memory controllers, while
striping every tensor makes each cross-node GPU consume remote pages. This was
rejected. A future engine could instead assign a whole tensor to a GPU and
place that tensor's pages on the GPU-local node.

### Per-GPU tensor-source placement

Assigning tensors to destination GPUs before upload and sourcing each from
GPU-local pages is the strongest long-term design. It requires engine changes
to page placement or replica selection and hardware performance validation.
That work is intentionally separated from this TUI setup simplification.

## Testing

Add deterministic tests for:

- GPU PCI bus and NUMA discovery, including sparse node IDs, `-1`, inaccessible
  nodes, malformed output, and missing tools;
- Fastest GPU staging on a four-node/two-GPU fixture selecting only the two
  GPU-local nodes;
- one shared copy and one engine regardless of GPU count;
- full-copy admission and fallback to the largest safe partial candidate;
- a missing or incompatible profile producing a blocker;
- unavailable GPU/CUDA discovery falling back to Single RAM copy with an
  explanation;
- Auto never selecting `per-node`;
- explicit Multiple NUMA replicas preserving multiplied copies and engines;
- managed launch and benchmark sanitizing ambient accelerator variables and
  applying only persisted reviewed values;
- `COLI_MMAP=1` and `COLI_RAMMAP=0` in the GPU-staging environment;
- preset selection, direct review navigation, Custom state, and existing
  manifest bypass in both Textual and curses frontends; and
- JSON/state round trips for the managed accelerator contract.

Run the focused planner, discovery, presentation, lifecycle, Textual, and
curses test modules, then the repository's full Python and C checks.

Hardware validation on the target system should record, for both the previous
shared defaults and the GPU-aware preset:

- startup VRAM-tier upload time;
- reported H2D time;
- GPU-to-NUMA mapping;
- staged page allocation;
- time to first token; and
- steady-state tokens per second.

The TUI must describe the result as topology-based optimization until measured
hardware data demonstrates a stronger performance claim.

## Acceptance criteria

- A new workspace presents exactly one intent-level question before review.
- Fastest GPU staging is the default and produces one shared copy and one
  multi-GPU engine.
- On the target topology, only the two GPU-local NUMA nodes provide staged
  pages and CPU affinity.
- Full staging is chosen only when the authoritative planner admits it;
  otherwise a compatible profile drives the largest safe partial staging plan.
- Auto never creates multiple model copies.
- Managed start uses the reviewed GPUs and GPU mmap upload path instead of
  silently stripping accelerator configuration.
- Existing manifests skip preset selection and remain unchanged.
- Advanced fields remain editable and visibly convert the draft to Custom.
- Both TUIs produce the same authoritative plan for the same preset and fixture.
- Focused tests and the full repository checks pass.
