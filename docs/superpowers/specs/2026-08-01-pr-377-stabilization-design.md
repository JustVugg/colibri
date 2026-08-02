# PR 377 stabilization and reconstruction design

Date: 2026-08-01

## Decision

PR #377 is not a merge candidate in its current 38-commit, 87-file form. The
current branch will receive only forward fixes until the complete CI matrix is
green. That green commit becomes an immutable donor for three smaller,
dependent pull requests reconstructed from reviewed hunks.

The audited range is pinned to base
`06f31b243792e02174a0c343021e5543d5072be4` and remote head
`3f795a7aa7af31a74abc6630664f2c1de8d96cb3`. Moving the reconstruction base
requires a separate compatibility review.

## Stabilization scope

The donor branch receives five focused change clusters:

1. Replace synthetic global platform mutation with injected platform services;
   make optional facade, frontend, benchmark, and HTTP-monitoring dependencies
   lazy.
2. Make wheel builds isolated and diagnostic, then apply the packaging fix
   supported by the exposed backend error. Test both the declared minimum and a
   current supported setuptools backend.
3. Separate portable tests from Linux operational tests, repair Windows and
   Darwin fixtures, use binary file descriptors on Windows, and make POSIX-only
   process control explicitly unsupported elsewhere.
4. Make mount and managed-process rollback fail closed and cover every recovery
   transition with fault injection.
5. Fix the source-level FP8 compiler diagnostic, align the CI matrix and skip
   inventory, and capture expected negative-test diagnostics.

No new product behavior, frontend features, benchmark policies, or unrelated
refactors belong in stabilization.

## Cleanup safety model

Mount ownership is persisted as `pending` before the mount helper runs. A mount
becomes managed only after its identity is recorded. Unreadable, replaced,
foreign, or nested mounts are never unmounted solely by pathname; they remain in
durable error state for explicit recovery.

Process absence must be positively established. An inconclusive identity check
cannot erase a termination failure when a retained direct-child handle still
reports the process alive. No usage merge, `stopped_at` publication, `ready`
restoration, or process-record removal occurs while absence is unproven. An
unverified process group is never signalled.

Cleanup may intentionally retain a resource after kernel refusal or unverifiable
ownership. The contract is no unsafe signal or unmount, no false clean result,
no premature accounting merge, and durable recovery metadata for every retained
resource.

## Supported headless contract

The replacement core is CLI-first:

- `plan --json`
- `stage --plan-token TOKEN --yes --json` (`prepare` may remain an alias)
- read-only `verify --json`
- `status --json`
- `destroy --deployment-token TOKEN --yes --json`

Plan and deployment tokens bind mutations to reviewed state. Result and error
schemas are versioned. Public managed `start` and `stop` remain deferred until
managed-runner recovery tests pass.

Neither current TUI is part of the mergeable replacement stack. If later demand
justifies a UI, one Textual companion may consume only the stable subprocess/JSON
contract; curses will not be maintained alongside it.

## Replacement pull requests

### PR1: engine RAMMAP, NUMA, and telemetry

Contains the engine C changes, CUDA accounting, strict telemetry protocol
parsing, focused documentation, and their tests. It excludes planning, mounts,
lifecycle, benchmark, and UI code.

### PR2: headless planning, staging, mounts, and recovery

Contains planning and discovery, platform operations, durable state, safe mount
recovery, the reduced facade and CLI, packaging, and portable tests. It depends
on PR1's engine/environment contract and must build with benchmark and UI modules
physically absent.

### PR3: managed runner, benchmark, and evidence

Starts as a draft and contains safe process supervision, a narrow deterministic
benchmark protocol, raw evidence production, and no frontend. A benchmark-only
commit boundary is preserved so optional UI work can never block or contaminate
the benchmark deliverable.

Shared files are assigned hunk by hunk to their earliest consumer. The donor's
cross-cutting commits are not cherry-picked wholesale.

## Measurement design

The causal CPU matrix holds the expert set and numeric residency budget fixed:

- anonymous PIN storage, interleaved across nodes 0 and 1;
- anonymous PIN storage, local to node 0;
- tmpfs RAMMAP storage, interleaved across nodes 0 and 1;
- tmpfs RAMMAP storage, local to node 0.

SSD-slab and tmpfs-slab controls separate media effects from direct mapping.
CUDA is evaluated in a separate block with fixed host and GPU budgets because
the production managed-CUDA policy changes RAMMAP and pinning simultaneously.
`PIN_GB=all` and `CUDA_EXPERT_GB=auto` are production-policy validation, not
causal evidence.

Each measured cell uses at least seven randomized fresh processes and records
the actual applied policy, topology, binary/model fingerprints, output hashes,
physical reads, swap, file/anonymous/shmem accounting, NUMA placement, DRAM
traffic, throughput, and latency. Staging is excluded from the measured engine
interval.

Correctness requires exact output parity, zero swap growth, zero physical SSD
reads for full RAMMAP runs, and verified requested placement. Performance claims
require predeclared practical thresholds and paired confidence intervals;
otherwise the result is reported as neutral.

## Verification and delivery gates

The donor is green only when Python, Linux, Windows/UCRT64, macOS, Nix Linux,
and Nix macOS pass at one commit, while real-tmpfs and zero-SSD-read checks remain
green. Wheel failures expose backend diagnostics, the platform skip inventory is
stable, and `git diff --check` passes.

Required lifecycle tests cover mount success followed by identity failure,
single- and multi-mount rollback, forced unmount failure, manifest-write failure,
live-child termination failure with inconclusive identity, proven-not-running
control, repeated interruption, forked-child readiness failure, and
foreign/replaced/nested mounts.

After the donor is green it is frozen. The three successor PRs are opened and
linked before #377 is closed as superseded. The existing branch is never rebased
or force-pushed.
