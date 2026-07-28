# RAM-workspace TUI

`coli ramdisk` is Colibri's guided Linux console for planning a NUMA-aware
tmpfs workspace, staging model weights, running managed engines, and comparing
storage paths. The interface is a view over the same lifecycle used by the
scriptable `plan`, `prepare`, `start`, `status`, `benchmark`, `stop`, and
`destroy` actions.

The TUI does not make an unsupported plan safe. It keeps the planner's
blockers, warnings, copy count, memory cost, CPU masks, mount paths, and
endpoints visible so that the operator can review the exact contract before
anything is mounted.

For a task-oriented walkthrough of one shared, full-model deployment, follow
[Run a shared full-model RAM workspace](ramdisk-tui-howto.md). This page
remains the complete reference for controls, lifecycle semantics, additional
modes, safety, and development.

## Requirements and platform scope

The RAM-workspace lifecycle and both terminal frontends are Linux-only. Other
Colibri commands remain portable, but `coli ramdisk` exits before opening a
TUI on macOS or Windows.

You need:

- a canonical Colibri model directory on durable storage;
- a terminal of at least 72 columns by 24 rows for the Textual interface;
- enough memory for the staged set, runtime overhead, and the planner's
  operating-system reserve;
- `sudo` authorization, or an existing root session, for `mount` and `umount`;
- Linux NUMA and cgroup information visible to the invoking process.

Install the supported Textual frontend from a checkout:

```sh
python3 -m pip install -r c/requirements-tui.txt
```

An unpacked release archive carries the same file at its top level. The Nix
package includes Textual in its pinned Python environment.

## Launching the console

Pass initial planner settings on the command line, then omit the lifecycle
action. The source-checkout examples below assume `cd c`; installed copies can
invoke `coli` from `PATH` instead:

```sh
./coli ramdisk --model /models/glm
./coli ramdisk --model /models/glm --mode partial --capacity-gb 96
./coli ramdisk --model /models/glm --memory-nodes 0,2 --cpu-list 0-31,64-95
```

`COLI_RAMDISK_UI` selects the frontend:

| Value | Behavior |
|---|---|
| `auto` | Default. Use Textual when installed; otherwise use curses. |
| `textual` | Require Textual. A missing dependency is reported instead of falling back. |
| `curses` | Use the compatibility curses interface explicitly. |

For example:

```sh
COLI_RAMDISK_UI=textual ./coli ramdisk --model /models/glm
COLI_RAMDISK_UI=curses ./coli ramdisk --model /models/glm
```

An invalid selector fails before terminal setup. `auto` falls back only when
Textual itself is missing; an unrelated import or startup failure is not
silently hidden.

## The placement contract

When no prepared workspace exists, both terminal interfaces begin with one
preset question:

- **Fastest GPU staging** (default) discovers NVIDIA GPUs, keeps one shared
  model copy and one engine, and selects only the effective NUMA nodes local to
  those GPUs. It first attempts full staging, then uses a compatible usage
  profile to fit a partial plan. The reviewed engine contract uses CUDA mmap,
  asynchronous RAM-to-VRAM copies, and automatic VRAM sizing.
- **Single copy RAM** keeps one shared copy and engine using the ordinary
  effective NUMA placement.
- **Minimal** requests a profile-guided partial plan sized to the largest
  safely admitted shard closure. Without a compatible usage profile it
  produces a blocked review with instructions instead of silently choosing
  another layout.
- **Multiple copies RAM** explicitly selects one complete copy and independent
  engine per chosen node.

If GPU discovery, PCI-to-NUMA locality, or the CUDA engine capability cannot be
used safely, Fastest GPU staging falls back visibly to the single-copy plan. It
never chooses replicas. A prepared workspace skips the preset question and
opens with its persisted placement. Selecting a preset jumps directly to
Review; the advanced steps remain editable, and an edit marks the preset as
Custom.

The rail at the top of the Textual interface summarizes the consequences of
the current plan:

```text
MODEL ×1  ──  RAM [N0 │ N2]  ──  ENGINE ×1  ──  PORT 8000
```

The normal `interleaved` topology means one staged copy and one managed engine.
Pages are placed across the selected NUMA memory nodes. Selecting more CPUs
does not multiply the model.

`per-node` topology means one complete staged copy and one independent engine
per selected node. It is replication, not model sharding, so the TUI marks it
as a danger state and shows the multiplied RAM and endpoint count. It is
available only through the explicit Multiple copies RAM startup choice or the
`--topology per-node` CLI option; automatic placement never enables it. The
interface can switch such a draft back to shared placement.

Multiple tmpfs mounts are independent filesystems, not a RAID stripe. The GPU
preset therefore uses one shared interleaved source over the GPU-local nodes
instead of striping mounts. With GPUs attached to different NUMA domains, a
single shared tensor can still have pages remote from one GPU; exact per-GPU
tensor placement requires engine-level placement rather than extra ramdisks.

Memory and CPU selections use Linux range-list syntax. The planner validates
memory nodes against the effective NUMA mask and CPU selections as complete
physical-core sibling groups inside the effective cpuset. DIMM and channel
inventory is informational; Linux memory policy is expressed in NUMA nodes.

## Textual workflow

The six numbered steps form one plan-and-operate workflow. The placement rail
remains visible throughout.

1. **Inspect** shows the canonical model, selected shard count, host memory,
   effective cores, NUMA nodes, per-node capacity, CPU lists, and NUMA
   distances.
2. **Placement** edits memory nodes, whole-core CPUs, and the managed mount
   root. It shows whether any selected CPUs imply remote-memory access.
3. **Capacity** chooses full or profile-guided staging and shows staged RAM,
   runtime reserve, current headroom, blockers, and warnings.
4. **Runtime** edits the base port, copy-worker count, context length, prefault
   setting, tmpfs huge-page policy, and whether swappable tmpfs is allowed.
5. **Review** repeats the exact copies, total RAM, engines, ports, nodes, CPU
   masks, and mount paths. Prepare opens a token-bound confirmation that
   expires after ten seconds.
6. **Operate** shows verified lifecycle actions and explains why unavailable
   actions are disabled.

Submit an input with Enter, or navigate away from the step. Navigation commits
valid pending input and rebuilds the authoritative plan. Invalid input keeps
you on the current step and leaves lifecycle actions disabled while the plan
is stale.

Once a workspace is prepared, its weight-placement settings are read from the
persisted manifest and remain locked until Destroy. A base port may be changed
while the workspace is ready or stopped, but not while managed engines are
running. The console refreshes status automatically; use a deep refresh when
you need hardware, source-fingerprint, and NUMA namespace validation repeated.

### Textual controls

| Control | Action |
|---|---|
| `Left` / `Right` | Previous or next step. |
| `1` through `6` | Open Inspect through Operate. |
| `r` | Deep-refresh hardware, model plan, and lifecycle validation. |
| `c` | Request cancellation of Prepare, Start, or Benchmark. |
| `q` or `Esc` | Request a safe quit. In an input, ordinary text keys remain input. |
| `Ctrl-C` | Request cancellation and exit with interrupt status after cleanup. |
| Tab / Shift-Tab | Move focus using Textual's standard focus navigation. |
| Enter / Space | Submit an input or activate the focused control. |

Inside a Prepare or Destroy review, `Esc` or `q` cancels only the review.
Resizing below 72 × 24 also closes an open review, and no action is taken.

## Lifecycle behavior

The buttons are state-driven; unavailable actions include a reason rather than
depending on the operator to remember the valid order.

### Plan

Before preparation, every settings change produces a new draft plan. Planning
scans model metadata and host constraints but does not mount, copy, or launch
anything. `NOT READY` blockers must be resolved before Prepare is enabled.

For automation or a complete versioned report, use:

```sh
./coli ramdisk plan --model /models/glm --json
```

### Prepare

Prepare mounts the reviewed tmpfs layout, copies the selected weight files,
builds the weights namespace, validates placement, and persists the manifest.
It does not start an engine.

The confirmation is bound to the exact plan. If settings change after review,
the action is refused and must be reviewed again. Copy progress remains visible
while navigation stays available.

### Start and status

Start launches only the persisted deployment. Shared placement starts one
managed engine on the base port. Per-node replication starts one engine per
mount, on `base port + NUMA node id`. Startup revalidates the effective CPU and
memory-node masks and refuses to widen beyond the reviewed whole-core masks.

Status distinguishes an absent, ready, starting, running, stopped, or
incomplete/error workspace. A fast status check verifies current mounts and
managed process identities. Deep refresh also checks the source fingerprint
and staged namespaces.

The scriptable equivalent is:

```sh
./coli ramdisk start --base-port 8000
./coli ramdisk status --json
```

### Benchmark

Benchmark is available only for prepared weights with managed engines stopped.
It runs deterministic, equal-control comparisons of the applicable persistent
SSD, tmpfs/slab, and direct RAM-map paths, then saves the scorecard and the
best safe runtime knobs. A subsequent Start can reuse those saved knobs.

Full-mode reports require valid physical-read accounting before claiming zero
SSD reads. Partial mode includes SSD fallback because unstaged weights remain
canonical on durable storage.

### Stop

Stop terminates only process groups whose persisted identity still matches the
managed engine. It leaves the RAM weights mounted, so the workspace can be
benchmarked or started again without restaging.

### Destroy

Destroy requires managed engines to be stopped. Its ten-second review is bound
to the current manifest and exact mount records; a changed deployment
invalidates the confirmation. It unmounts the volatile weight workspace but
preserves durable engine state and benchmark history.

Scriptable cleanup is:

```sh
./coli ramdisk stop
./coli ramdisk destroy --yes
```

## Full and partial staging

**Full mode** stages every model shard. Its default capacity is the full model
size, and managed full mode enables direct-map prefaulting by default. This is
the path used to verify that full-resident runs avoid physical SSD weight
reads.

**Partial mode** requires a positive per-copy `--capacity-gb` budget and a
compatible usage profile. If `--profile` is omitted, the planner looks for
`<model>/.coli_usage`. It selects complete safetensors shard closures within
the budget; the TUI shows both profile coverage and shard-granularity staging
efficiency. Unselected canonical weights stay on durable storage and remain
available through the staged namespace.

You need the real model to stage it or run real-model benchmarks and engines.
You do not need the model for the normal build, unit, frontend, packaging, or
generated-fixture lifecycle tests.

## Privilege and safety model

Planning and status are unprivileged. Before Prepare or Destroy, the Textual
interface temporarily suspends itself for one foreground `sudo -v` prompt.
It then verifies that the credential can be reused non-interactively. The
background worker uses non-interactive sudo only for the exact mount or
unmount commands, and a keepalive preserves rollback authority during a long
copy. If the credential cannot be reused without prompting, no mount operation
starts.

Other safety boundaries include:

- the mount root must be a non-symlink path below `/mnt`;
- an existing mount root must satisfy the ownership and emptiness checks;
- private tmpfs mounts use mode `0700`;
- Colibri never runs `swapoff` or changes host-global HugeTLB or weighted
  interleave settings;
- swappable tmpfs on kernels without `noswap` support requires an explicit
  opt-in;
- model files and durable `.coli_usage` / `.coli_kv*` state are not modified
  by staging;
- lifecycle mutations are serialized with a per-user lock;
- mount, process, plan, and confirmation identities are revalidated before
  destructive or process-control actions.

Use a cgroup or cpuset when strict host-level isolation is required. The
planner treats the tighter of host/NUMA availability and cgroup memory
headroom as the admission boundary.

## Cancellation and cleanup

Prepare, Start, and Benchmark are cooperatively cancellable. Press `c`, use the
visible Cancel button, request quit, or send `SIGHUP`/`SIGTERM`; Colibri waits
for a rollback or cleanup checkpoint before returning control. A cancelled
Prepare rolls back mounts it created rather than leaving an untracked
workspace.

Stop and Destroy are verified cleanup transactions and are deliberately not
interruptible halfway through. A quit or termination request waits for them to
finish. If cleanup itself fails, the error remains visible and automatic quit
is cancelled so the operator can inspect the workspace.

## Durable state

By default the lifecycle stores private state under:

```text
~/.local/state/colibri/ramdisk/
├── manifest.json
├── benchmarks.json
├── lifecycle.lock
├── engines/
└── benchmarks/
```

If `XDG_STATE_HOME` is set, the root is
`$XDG_STATE_HOME/colibri/ramdisk`. It must be an absolute path on durable
storage. tmpfs/ramfs paths, symlink-containing private state paths, and paths
inside the volatile weight mount are rejected.

`COLI_RAMDISK_MANIFEST` can override only the manifest path; it must also be
absolute and durable. Set these variables before the first lifecycle action
and keep them consistent for every later invocation.

Destroy removes the volatile mounts and manifest but preserves durable engine
KV/usage state and benchmark history.

## Curses compatibility interface

The curses frontend exposes the same lifecycle and safety rules through five
pages: Plan, Hardware, Activity, Benchmark, and Settings. Press `?` for its
built-in help.

| Key | Action |
|---|---|
| `Left` / `Right` or `h` / `l` | Change page. |
| `Up` / `Down` or `j` / `k` | Scroll; Page Up and Page Down move a viewport. |
| `p` | Review and prepare from the Plan page; press again within ten seconds to confirm. |
| `s` / `x` / `d` | Start, stop, or review/destroy. Destroy also requires a second press within ten seconds. |
| `b` | Benchmark from the Benchmark page. |
| `R` | Deep refresh. |
| `c` | Cancel a cancellable active operation. |
| `?` | Open or close help. |
| `q`, `Esc`, or `Ctrl-C` | Request a safe quit. |

Settings-page shortcuts are displayed beside each value. In particular, `m`
switches full/partial mode, `i` switches an explicitly requested replica draft
back to shared placement, and uppercase `P` edits the base port. The curses
frontend accepts a smaller 38 × 8 terminal, although more space is useful for
reviewing the full contract.

The curses interface is a compatibility fallback. Textual provides labeled
controls, keyboard focus navigation, explicit textual state labels, and
viewport guards. Scriptable lifecycle actions remain the stable
noninteractive fallback when a terminal UI is unsuitable.

## Troubleshooting

**The command says the TUI is Linux-only.**

The portable engine and CLI work on other supported platforms, but tmpfs/NUMA
lifecycle operations do not. Run planning and staging on a Linux host.

**Textual was requested but is not installed.**

Install `c/requirements-tui.txt`, use the Nix package, or select
`COLI_RAMDISK_UI=curses`.

**The terminal shows only a resize notice.**

Resize Textual to at least 72 × 24. The notice is intentional: confirmation
controls are hidden when the complete contract cannot be reviewed safely.

**Prepare is disabled.**

Read every blocker on Capacity. Common causes include insufficient host or
cgroup memory, an incompatible partial profile, an unsafe mount root, an
invalid NUMA/CPU range, incomplete physical cores, or an already persisted
workspace.

**Settings are locked.**

Prepared weight settings are immutable. Stop any running engines, then Destroy
the workspace before changing placement or staging settings. The base port can
be edited while ready or stopped.

**Benchmark is disabled.**

Prepare the workspace first and stop all managed engines. Benchmark refuses to
share the fixed environment with a serving process.

**Destroy is disabled.**

Stop managed engines first. If process cleanup is pending, Stop remains
available even for an incomplete workspace.

**Sudo authorization returns to the TUI without mounting.**

The foreground prompt may have been cancelled, or policy may forbid
noninteractive reuse. Colibri will not allow a background password prompt.
Authorize a reusable ticket under the applicable sudo policy and retry.

**Refresh reports that masks changed.**

The invoking process's cgroup/cpuset CPU or memory-node allowance differs from
the prepared contract. Restore the original boundary or stop and destroy the
workspace, then prepare a new reviewed plan.

**A refresh failed.**

No lifecycle action is taken on a refresh error. Press `r` in Textual or `R`
in curses after correcting the reported filesystem, model, permission, or
host-state problem.

## Testing and development

The frontend is intentionally separated from lifecycle authority:

- `c/ramdisk_textual.py` renders the guided interface and delegates every plan
  rebuild and mutation;
- `c/ramdisk_ui.py` contains dependency-free placement, health, action-policy,
  and review projections;
- `c/ramdisk_support/curses_ui.py` contains the compatibility frontend;
- `c/ramdisk_support/cli.py` selects a frontend lazily;
- `c/ramdisk.py` is the compatibility facade over the planning, state, mount,
  process, lifecycle, and benchmark support modules.

The ordinary gate is model-free and unprivileged:

```sh
make -C c check
```

To run the Textual pilot tests directly, install the TUI dependency and run:

```sh
cd c
python3 -m unittest discover -s tests -p 'test_ramdisk_textual.py' -v
```

These tests drive the interface in a virtual terminal. They cover backend
selection, plan rebuilding, action guards, exact reviews, viewport limits,
replication warnings, settings locks, and cooperative cancellation without
mounting tmpfs or loading a model.

The opt-in real-tmpfs integration test uses a generated tiny model fixture, so
it exercises Prepare, Status, and Destroy without downloading GLM-5.2:

```sh
cd c
COLI_RAMDISK_INTEGRATION=1 \
  python3 -m unittest discover -s tests -p 'test_ramdisk_integration.py' -v
```

Run it only in a suitable privileged or private mount environment. Its cleanup
path is part of the test.

The RAM-map end-to-end test is separate and consumes an existing compatible
tmpfs-backed model supplied by the caller:

```sh
cd c
COLI_RAMMAP_E2E_MODEL=/dev/shm/glm_i4 \
  python3 -m unittest discover -s tests -p 'test_rammap_e2e.py' -v
```

It does not download a model. Neither opt-in test is part of the normal
cross-platform gate.
