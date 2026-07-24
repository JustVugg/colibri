# Cross-Platform RAM-Disk Refactor Design

**Status:** Approved for implementation  
**Date:** 2026-07-23  
**Scope:** Pull request #377 review feedback, cross-platform testability, and maintainability of the RAM-disk control plane

## Context

The RAM-disk control plane is implemented primarily in `c/ramdisk.py`, which is
7,419 lines and currently combines:

- durable state and manifest handling;
- cgroup, memory, CPU, and NUMA discovery;
- safetensors model scanning and partial-staging selection;
- capacity and placement planning;
- privileged tmpfs mount and rollback operations;
- shard copying and namespace validation;
- managed process identity, launch, signaling, and cleanup;
- benchmark execution and scoring;
- CLI parsing and human-readable output; and
- the legacy curses interface.

The feature itself depends on Linux facilities such as tmpfs mount policies,
`/proc`, NUMA masks, process groups, and pidfds. It is therefore intentionally
Linux-only. The Python module and its tests are still discovered on Windows and
macOS by the repository-wide `make check` gate. Those tests currently invoke
POSIX- and Linux-only APIs directly, producing failures such as missing
`os.statvfs`, `os.getuid`, `os.geteuid`, `os.killpg`, and `/proc`.

On macOS, `tempfile` commonly returns a path below `/var`, which resolves through
the system `/var -> /private/var` symlink. This collides with the control plane's
intentional policy of rejecting symlinked durable-state components.

The repository has a committed `flake.nix` and `flake.lock` providing Linux and
macOS packages and a development shell. It has no Devbox configuration. Native
Windows builds and tests use MSYS2/UCRT64 in GitHub Actions.

## Goals

1. Keep the actual RAM-disk lifecycle Linux-only and fail clearly elsewhere.
2. Run portable planning, schema, validation, CLI, and package tests on Linux,
   macOS, and native Windows.
3. Prevent portable tests from accidentally invoking unavailable operating
   system APIs.
4. Preserve the strict durable-state symlink policy.
5. Split `c/ramdisk.py` into modules with one clear responsibility and an
   acyclic dependency direction.
6. Preserve the existing public Python functions, JSON schemas, manifest
   format, CLI behavior, and flat-release usability.
7. Use Nix for reproducible Linux/macOS coverage and MSYS2 for Windows without
   introducing a second environment manager.

## Non-Goals

- Implementing native Windows RAM-disk lifecycle support.
- Implementing a macOS RAM-disk lifecycle backend.
- Changing the manifest, plan, status, or benchmark schema versions.
- Replacing existing dictionaries with a new dataclass/domain-object model.
- Redesigning the Textual interface.
- Adding Devbox solely as a wrapper around the existing Nix flake.
- Changing runtime behavior unrelated to the review feedback.

## Chosen Architecture

`c/ramdisk.py` remains as a small compatibility facade. Implementation moves to
an internal `c/ramdisk_support/` package:

```text
c/ramdisk.py
c/ramdisk_support/
├── __init__.py
├── common.py
├── platform_ops.py
├── linux_ops.py
├── state.py
├── discovery.py
├── model.py
├── planning.py
├── mounts.py
├── processes.py
├── lifecycle.py
├── benchmark.py
├── presentation.py
├── curses_ui.py
└── cli.py
```

Responsibilities:

- `common.py`: shared exceptions, constants, timestamps, range parsing, and
  dependency-free utilities.
- `platform_ops.py`: the capability interface used by portable orchestration,
  plus an unsupported-platform implementation that returns explicit
  capability failures.
- `linux_ops.py`: Linux implementations for `/proc`, cgroups, NUMA, filesystem
  capacity, process groups, pidfds, mount inspection, and privileged commands.
- `state.py`: state-root derivation, lifecycle locking, atomic JSON,
  manifest validation, durable path checks, and usage-history merging.
- `discovery.py`: hardware/cgroup discovery and normalized capability reports.
- `model.py`: safetensors scanning, fingerprints, profiles, shard closures,
  and model-size calculations.
- `planning.py`: topology, placement, capacity, reserve, and partial-staging
  decisions.
- `mounts.py`: mount options, privilege handling, staging copies, validation,
  cancellation, and rollback.
- `processes.py`: process identity, managed child tracking, readiness,
  admission, signaling, and cleanup primitives.
- `lifecycle.py`: `prepare`, `start`, `stop`, `destroy`, and `status`
  orchestration.
- `benchmark.py`: benchmark engines, variants, aggregate execution, metrics,
  scorecards, and report persistence.
- `presentation.py`: human-readable plans, reports, confirmations, and
  shared view-row construction.
- `curses_ui.py`: the existing curses frontend and its worker/signal handling.
- `cli.py`: parser configuration, dispatch, JSON output, and command-level
  termination guards.

The dependency direction is:

```text
common
  ├── platform_ops -> linux_ops
  ├── state
  ├── discovery
  └── model
        └── planning
              ├── mounts
              ├── processes
              └── lifecycle
                    ├── benchmark
                    ├── presentation
                    ├── curses_ui
                    └── cli
```

Lower layers must not import CLI or UI modules. UI modules receive data through
the existing plan/status/report dictionaries and action callbacks.

## Compatibility Facade

`c/ramdisk.py` continues to export the current externally used surface:

- `RamdiskError`
- `discover_hardware`
- `scan_model`
- `build_plan`
- `configure_parser`
- `prepare`
- `start`
- `stop`
- `destroy`
- `status`
- `benchmark`
- `dispatch`
- `launch_tui`

The facade may temporarily re-export selected private helpers used by existing
tests while those tests move beside their owning modules. No production caller
should gain a dependency on new private modules.

The facade remains importable on every supported Python platform. Importing it
must not probe `/proc`, import `fcntl` unconditionally, or resolve Linux system
binaries. Linux capabilities are selected lazily when a lifecycle action is
requested.

## Platform Capability Boundary

Portable code must not directly call:

- `os.getuid` or `os.geteuid`;
- `os.killpg`, `os.pidfd_open`, or `signal.pidfd_send_signal`;
- `os.statvfs`;
- `/proc`, `/sys`, or cgroup files;
- `mount`, `umount`, or `sudo`; or
- Linux NUMA policy interfaces.

Those operations live behind narrow capability methods. Tests inject a fake
implementation rather than patching attributes that may not exist on the host.

On Windows and macOS:

- parsing, model scanning, plan schema construction, and presentation remain
  usable with injected hardware fixtures;
- `--help`, packaging, imports, and JSON error contracts work normally; and
- real lifecycle actions return the existing Linux-only blocker before any
  unsupported operation is attempted.

The platform boundary is intentionally internal. It is not a promise of native
RAM-disk support on non-Linux hosts.

## Durable-State and Symlink Policy

Production durable-state paths remain symlink-free after normalization. The
manager must not silently follow a redirected component before `makedirs`,
`chmod`, manifest writes, usage merges, or process cleanup.

Tests must distinguish between:

1. an operating-system-provided temporary root whose spelling is non-canonical;
2. an explicitly configured state root; and
3. a symlink inserted inside a manager-owned state path.

Cross-platform fixtures resolve their temporary root once before deriving
`XDG_STATE_HOME`, manifest, model, or engine-state paths. Negative tests create
an explicit redirected component below that canonical root and continue to
assert rejection without chmodding the target.

This fixes macOS fixture behavior without weakening the Linux security
invariant.

## Testing Strategy

Tests are organized by responsibility rather than retained in a single
three-thousand-line test module:

```text
c/tests/
├── test_ramdisk_common.py
├── test_ramdisk_state.py
├── test_ramdisk_discovery.py
├── test_ramdisk_model.py
├── test_ramdisk_planning.py
├── test_ramdisk_mounts.py
├── test_ramdisk_processes.py
├── test_ramdisk_lifecycle.py
├── test_ramdisk_benchmark.py
├── test_ramdisk_cli.py
├── test_ramdisk_packaging.py
└── test_ramdisk_integration.py
```

Test tiers:

- **Portable unit tests:** run on Linux, macOS, and Windows. They cover pure
  calculations, schemas, parsing, state transformations, injected platform
  behavior, CLI output, and packaging.
- **Linux backend contract tests:** use fake filesystem/process inputs where
  possible and run only on Linux when the test exercises real Linux APIs.
- **Privileged integration tests:** remain Linux-only and opt-in unless the CI
  job explicitly provisions the required mount privileges.
- **Unsupported-platform tests:** patch the selected platform backend and
  verify that lifecycle actions fail before OS calls.

Before moving implementation, characterization tests freeze:

- public facade exports;
- JSON schema names and versions;
- plan/status/benchmark key sets;
- confirmation tokens;
- manifest compatibility; and
- parser/dispatch behavior.

## Nix, Windows, and CI

Nix remains the reproducible environment for Linux and macOS:

- keep `flake.lock` committed;
- include the new `ramdisk_support` package in the derivation output;
- ensure the development shell contains the tools needed for `make check`;
- run portable tests on both Linux and Darwin Nix systems; and
- run Linux backend tests only when `stdenv.hostPlatform.isLinux`.

Native Windows continues to use MSYS2/UCRT64 because Nix does not provide the
project's claimed native Windows environment.

GitHub Actions should provide:

1. existing `make check` jobs on Linux, Windows, and macOS;
2. a Nix build/check job on Linux;
3. a Nix build/check job on macOS if runner cost and dependency availability
   are acceptable; and
4. the existing explicit Linux real-tmpfs integration job.

Devbox is not added. If maintainers later want it, it should be a thin,
separately reviewed convenience layer rather than a second source of package
versions.

## Packaging

Every supported installation layout must include `ramdisk_support/`:

- source checkout;
- editable/wheel installation through setuptools;
- `make install`;
- release archives;
- Nix derivation; and
- test packaging fixtures.

`c/coli` continues to locate one complete support bundle. Its completeness check
must validate the package directory and required module files atomically rather
than accepting a partial mixture of old `ramdisk.py` and new support modules.

Packaging tests must import the facade and exercise `coli ramdisk --help` from
an isolated installed/archive layout.

## Migration Sequence

1. Rebase the feature branch onto current `JustVugg/dev` and resolve conflicts.
2. Add facade/schema characterization tests.
3. Create `ramdisk_support` and extract dependency-free common/model/planning
   code.
4. Introduce the capability interface and move Linux discovery into
   `linux_ops.py`.
5. Extract state and durable-path logic; canonicalize test fixture roots.
6. Extract mount/copy and process-management primitives.
7. Move lifecycle orchestration while preserving facade signatures and
   decorators.
8. Extract benchmark and presentation code.
9. Move curses code; retain `ramdisk_textual.py` as the existing separate
   frontend.
10. Split the monolithic test file alongside the implementation modules.
11. Update setuptools, release archives, `make install`, and Nix packaging.
12. Add/adjust cross-platform and Nix CI coverage.
13. Run the full verification matrix and check PR mergeability.

Each extraction is a mechanical, behavior-preserving commit where practical.
Behavioral portability fixes are kept separate from file-movement commits so
reviewers can distinguish code motion from semantic changes.

## Verification and Acceptance

Required local checks:

- focused unit tests for every extracted module;
- the complete Python test suite;
- C tests when rebasing or packaging changes touch engine/build files;
- `make check`;
- `git diff --check`; and
- isolated packaging/import smoke tests.

Required CI outcomes:

- Linux `make check`: pass;
- macOS `make check`: pass;
- Windows MSYS2/UCRT64 `make check`: pass;
- Linux Nix build/check: pass;
- macOS Nix build/check: pass or an explicitly documented external runner
  limitation; and
- PR reports no merge conflicts with `dev`.

The work is accepted when:

- no portable test reaches a missing host API;
- unsupported lifecycle calls fail with a stable Linux-only message;
- explicit symlink redirection is still rejected;
- all existing public functions and JSON schema versions remain compatible;
- release, wheel, `make install`, and Nix layouts load one complete support
  bundle; and
- `c/ramdisk.py` is a small facade rather than an implementation module.

## Risks and Mitigations

- **Circular imports:** enforce the dependency direction above and keep shared
  exceptions/constants in `common.py`.
- **Mock target breakage:** add facade compatibility exports temporarily, then
  move tests to patch the owning module.
- **Packaging omissions:** update all layouts in the same stage and verify each
  from an isolated directory.
- **Behavior drift during code movement:** freeze schemas/signatures first and
  separate mechanical extraction commits from portability changes.
- **Security regression:** keep path and identity tests as mandatory gates on
  every lifecycle extraction.
- **Oversized review:** use incremental commits and avoid unrelated dataclass,
  schema, or Textual redesigns.
