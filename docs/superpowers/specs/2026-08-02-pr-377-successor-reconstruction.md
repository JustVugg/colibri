# PR #377 successor reconstruction ledger

Date: 2026-08-02 (donor frozen 2026-08-03)

Uncommitted planning artifact. The donor branch is frozen as an immutable
source; PR #377 is never merged. Three smaller, dependent pull requests are
reconstructed hunk-by-hunk from the green donor, then #377 is closed as
superseded. Authoritative scope: `2026-08-01-pr-377-stabilization-design.md`.

## Green donor

- **Donor SHA:** `5f6f31a00b6dfd57f73484b046ed34081f520f92`
  - pushed to `origin/feature/ramdisk_streaming`; PR #377 head.
  - 5 forward-only repair commits on `321b9bc` (no rebase/squash/amend/force):
    `cf2f15b` (F1: retained managed-process absence proven globally before usage
    merge), `fc5f65b` (F2: multi-engine readiness-death + survivor rollback
    coverage), `02a7f94` (sentinel binary-mode test exercises the real stream
    primitive), `62dcea5` (correct stale `COLI_RAMMAP_E2E` gate docs),
    `5f6f31a` (clear stale recovery errors after verified stop; lock F2 order).
  - Local verification green: `make -C c check` (native C + Python 3.10, 878
    passed / 19 skips), Python 3.12 discovery (865 / 55 skips), forced
    Win32/no-dirfd lifecycle matrix on both runtimes, packaging tests on both
    runtimes, changed-file `py_compile` both runtimes, `yq` parse, `git diff
    --check`, and three wheel smokes (setuptools 77.0.1 minimum, current, and
    default isolated PEP 517).
  - Immutable-SHA sign-off via subagents at `5f6f31a`: **PASS**, zero
    in-scope survivors (5 scope-calibrated lane audits + adversarial verify +
    final judge). The one finding raised (LOW: `stop()` popping `recovery` on
    the stopped branch) was adversarially refuted — retained mounts force
    `state="error"`, so the "stopped + retained metadata" precondition is
    unreachable.
  - **CI status:** the four OS-specific lanes (Windows UCRT64, macOS, Nix
    Linux, Nix macOS) ran on upstream as `event=pull_request` for `5f6f31a`
    but are `action_required` (fork-PR workflow approval). They need an admin
    on `JustVugg/colibri` to approve them; approve before declaring the donor
    fully-CI-green. All locally-runnable lanes (committed-range, linux, both
    wheel-backend cells, wheel-isolated) are green.

- **upstream/dev base merged into the donor:** `72ddb673b454f5132482ebfffffa9faaa98ae070` (merged at `60ed23b`, forward-only).
- **Current `upstream/dev`:** `7fb11595245b6f320b3cfef9565fdbe6da92442b` (advanced
  since the merge). A forward re-merge of `upstream/dev` may be required before
  PR1 if `dev` diverges further; always integrate forward-only.

## Threat-model boundaries accepted (NOT donor blockers)

These were flagged by audits and independently adjudicated out of stabilization
scope; each maps to a successor deliverable below.

- **Setsid / env-shedding descendant escape** (ordinary stop, retained, and
  post-termination paths): the donor proves process absence within the
  cooperative UID+nonce attribution model (group gone AND zero
  nonce-attributable processes). A descendant that sheds *all* attribution
  (`setsid()` + `execve()` to a different binary with an empty environment) is
  outside that model and reduces to the out-of-scope hostile-same-UID ABA
  boundary. Positively tracking it requires kernel containment → **PR3**.
- **Strict telemetry protocol parsing** (10/17/18-field contract, finiteness,
  validity flag) → **PR1** (`serve_protocol.md` is currently descriptive).
- **Legacy headerless usage-history acceptance**: intended legacy journal/seed
  adoption (guarded by usage-transaction-ID / canonical-marker /
  parent-identity authority; covered by `test_legacy_*`). Header values that
  coerce to valid ints are benign.
- **Darwin `vm_stat` failure fallback to installed RAM**: baseline behavior
  preserved verbatim by stabilization commit `1eb29e3`.
- **runtime_monitor mid-sample identity revalidation**: monitoring/telemetry
  attribution, not cleanup safety → **PR3**.

## Three-PR dependency chain

`PR1 → PR2 → PR3`. PR2 depends on PR1's engine/environment contract; PR3
depends on PR2's durable state + recovery. Shared files are split hunk-by-hunk
to their earliest consumer (the donor's cross-cutting commits are not
cherry-picked wholesale).

### PR1 — engine RAMMAP, NUMA, and telemetry

Engine C changes, CUDA accounting, **strict telemetry protocol parsing**, and
their tests. Excludes planning, mounts, lifecycle, benchmark, UI.

- `c/colibri.c`, `c/inkling.c`, `c/kimi_k3.c`, `c/olmoe.c` (RAMMAP, NUMA
  pinning, CUDA accounting, in-place re-exec).
- `c/compat.h` (engine binary-mode primitive, incl. the `coli_serve_binary_mode_stream`
  refactor at `02a7f94`), `c/st.h`, `c/telemetry.h`, `c/route_trace.h`.
- `c/openai_server.py` — **tighten PROF parsing** (design 78-80): reject
  field counts other than 10/17/18; read `physical_ssd_valid` as `== "1"` not
  `bool(int(...))`; reject non-finite numerics; make `serve_protocol.md:91`
  normative.
- Tests: `c/tests/test_openai_server.py`, `c/tests/test_rammap_e2e.py` (parse
  logic; the live canonical/tmpfs cells require the gates below), C engine
  tests `c/tests/test_serve_sentinel.c`, `c/tests/test_ue8m0.c`.
- Docs: `docs/serve_protocol.md`, `docs/cuda.md`, `docs/inkling.md`, `docs/kimi_k3.md`.

### PR2 — headless planning, staging, mounts, and recovery

Planning/discovery, platform operations, durable state, safe mount recovery,
the **reduced facade + tokenized CLI**, packaging, and portable tests. Must
build with benchmark and UI modules physically absent.

- `c/ramdisk.py`.
- `c/ramdisk_support/{__init__,accelerator,cli,common,discovery,lifecycle,linux_ops,model,mounts,planning,platform_ops,presets,processes,state}.py`.
- Packaging/build: `pyproject.toml` (drop Textual hard dep), `c/coli` (launcher
  must not require UI modules), `flake.nix`.
- Portable tests: `c/tests/test_ramdisk_state_lifecycle.py`,
  `c/tests/test_ramdisk_processes.py`, `c/tests/test_ramdisk_planning_module.py`,
  `c/tests/test_ramdisk_packaging.py`, `c/tests/test_ramdisk_platform.py`.
- Docs/CI: `README.md`, `docs/{ENVIRONMENT,SETTINGS,api,quickstart,ramdisk-tui-howto,ramdisk-tui}.md`,
  `docs/superpowers/specs/2026-08-01-pr-377-stabilization-design.md`,
  `.github/workflows/{check,ci,release}.yml`.

**Tokenized JSON contract to implement here** (design 56-73); result/error
schemas are versioned:

- `plan --json`
- `stage --plan-token TOKEN --yes --json` (`prepare` may remain an alias)
- read-only `verify --json`
- `status --json`
- `destroy --deployment-token TOKEN --yes --json`

Plan/deployment tokens bind mutations to reviewed state. Move token
construction out of `presentation.py` (which imports `ramdisk_ui`) into a
headless core module; keep the existing `importlib` lazy UI accessors in
`ramdisk.py`. Public managed `start`/`stop` remain **deferred until PR3**.

### PR3 — managed runner, benchmark, and causal evidence

Safe process supervision, the narrow deterministic benchmark, raw evidence
production, no frontend. Starts as a draft. A benchmark-only commit boundary
is preserved so optional UI can never block/contaminate the benchmark.

- `c/ramdisk_support/{benchmark,runtime_monitor}.py`.
- Managed-supervision hunks of `c/ramdisk_support/lifecycle.py` and
  `processes.py` (the recovery absence-proof itself is PR2; **kernel
  containment** — cgroup-v2 / pidfd per-descendant — is PR3 and closes the
  setsid/env-shedding escape).
- Tests: `c/tests/test_ramdisk_benchmark.py`, live cells of
  `c/tests/test_rammap_e2e.py`.
- `runtime_monitor.py`: bind health/profile responses to deployment/nonce
  identity and revalidate ownership after sampling (closes the mid-sample
  identity gap). Managed `start`/`stop` JSON lands here.

## Shared / hunk-split files

- `c/compat.h` → PR1 (engine protocol).
- `c/ramdisk_support/lifecycle.py` → PR2 (recovery absence-proof, mounts,
  staging, stale-error clearing) **except** managed-supervision hardening → PR3.
- `c/ramdisk_support/processes.py` → PR2 (identity/`_process_matches`,
  readiness death branch) **except** supervision hunks → PR3.
- `c/ramdisk_support/presentation.py` → PR2 (relocate token core to a headless
  module); UI rendering excluded.
- `c/tools/clean.py` → PR2 (build artifact symmetry incl. `test_serve_sentinel`
  and `test_ue8m0`); other `c/tools/*` to their consumers.

## Physically no frontend/TUI in any successor

`ramdisk_ui.py`, `ramdisk_textual.py`, `curses_ui.py`, and the UI-rendering
portions of `presentation.py` are excluded from every successor. PR2 must build
and install with these physically absent: remove their mandatory bundle
membership in `c/coli`, drop Textual as a hard `pyproject.toml` dependency, and
rely on the lazy `importlib` accessors already in `ramdisk.py`. If a UI is
later wanted, one Textual companion may consume only the stable subprocess/JSON
contract; curses will not be maintained alongside it (design 70-72).

## Causal evidence acceptance criteria (PR3)

Fixed-expert, fixed-residency CPU matrix (design 99-123):

- anonymous PIN storage, interleaved across nodes 0 and 1;
- anonymous PIN storage, local to node 0;
- tmpfs RAMMAP storage, interleaved across nodes 0 and 1;
- tmpfs RAMMAP storage, local to node 0.

SSD-slab and tmpfs-slab controls separate media effects from direct mapping.
CUDA is a separate block with fixed host/GPU budgets (the production
managed-CUDA policy changes RAMMAP and pinning simultaneously). `PIN_GB=all`
and `CUDA_EXPERT_GB=auto` are production-policy validation, not causal evidence.

Each measured cell uses **at least seven randomized fresh processes** and
records: actual applied policy, topology, binary/model fingerprints, output
hashes, physical reads, swap, **file/anonymous/shmem accounting, NUMA
placement, DRAM traffic**, throughput, and latency. Staging is excluded from
the measured engine interval.

**Correctness** requires exact output parity, zero swap growth, zero physical
SSD reads for full RAMMAP runs, and verified requested placement. **Performance
claims** require predeclared practical thresholds and paired confidence
intervals; otherwise the result is reported neutral. The donor's current
benchmark does not implement this matrix (it removes inherited `PIN*`, derives
one NUMA policy, runs one process × three turns, and omits DRAM traffic and
file/anonymous/shmem accounting) — PR3 supplies a new experiment runner and
evidence schema; it must not claim evidence the donor does not produce.

## Open and cross-link all successors before closing PR #377

1. Branch PR1 off the green donor (re-merge `upstream/dev` forward-only first if
   `dev` has diverged past `72ddb673`).
2. Open PR2 on a branch depending on PR1's branch.
3. Open PR3 as a **draft** on a branch depending on PR2's branch.
4. Each PR description cross-links the other two, this ledger, and the
   `2026-08-01` stabilization design.
5. Only after all three are open and linked, **close PR #377 as superseded**
   (do not merge it). The donor branch is never rebased or force-pushed.
