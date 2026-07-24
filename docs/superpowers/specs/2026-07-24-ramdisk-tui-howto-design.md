# RAM-workspace TUI how-to and visual-capture design

## Context

The repository already contains `docs/ramdisk-tui.md`, a combined operator and
developer reference for the RAM-workspace Textual and curses interfaces. It
describes controls, lifecycle semantics, safety invariants, troubleshooting,
and test strategy, but it does not provide a visual, task-oriented walkthrough.

This design adds a separate Diátaxis how-to guide for Linux operators who are
comfortable in a terminal but new to the Colibri RAM-disk workflow. Its single
outcome is completing a shared, full-mode deployment on a host whose reviewed
plan has no blockers:

`launch → inspect → prepare → start/status → stop → destroy`

The guide uses screenshots from the real Textual application code. Screenshot
generation must never mount tmpfs, invoke sudo, launch an engine, mutate a
model, or depend on the authoring host having enough memory for GLM-5.2.

## Documentation classification

- **Document type:** How-to guide.
- **Audience:** Linux operators familiar with terminals but new to Colibri
  RAM-disk operation.
- **Reader goal:** Complete one safe shared, full-mode lifecycle on a
  sufficiently large Linux host.
- **Primary success condition:** The reader destroys the volatile workspace,
  verifies that no managed engines or mounts remain, and understands which
  durable state was preserved.

## Goals

1. Provide an exact, linear operator procedure instead of duplicating the
   existing reference guide.
2. Show the real Textual interface at the decision points where a screenshot
   materially reduces ambiguity.
3. Make the preconditions for Prepare explicit, especially memory reserve,
   whole-core CPU masks, NUMA placement, mount-root safety, and reusable sudo
   authorization.
4. Explain cancellation and rollback at the point where they matter.
5. Keep illustrative screenshot data clearly separate from commands and
   values the reader must verify on their own host.
6. Make screenshot generation deterministic and safe to run on ordinary
   development machines.
7. Use Playwright to validate the rendered documentation without adding it as
   an application runtime dependency.

## Non-goals

- Teaching profile-guided partial staging.
- Teaching per-node replication.
- Teaching RAM-disk benchmarking.
- Downloading or converting a model.
- Performance tuning or throughput claims.
- Replacing the complete control/keyboard reference in
  `docs/ramdisk-tui.md`.
- Capturing a live production host or representing illustrative values as
  measured hardware results.
- Changing TUI behavior or lifecycle policy for documentation convenience.

## Chosen documentation structure

The new guide will be `docs/ramdisk-tui-howto.md` with the following sections.

### 1. Before you begin

List the requirements that must be true before launching:

- Linux host;
- Textual dependency installed;
- compatible Colibri engine and canonical model directory on durable storage;
- terminal at least 72 columns by 24 rows;
- safe non-symlink mount root below `/mnt`;
- enough host, NUMA-node, and cgroup memory headroom;
- complete physical-core CPU selection; and
- sudo policy that permits foreground authorization followed by
  noninteractive reuse.

The section will state that the operator must not override a planner blocker.

### 2. Launch the RAM-workspace TUI

Provide source-checkout and installed-package commands with
`COLI_RAMDISK_UI=textual`. Explain that planning and inspection are
unprivileged.

Screenshot: initial Inspect step with shared placement.

### 3. Review a shared full-mode plan

Walk through Inspect, Placement, Capacity, Runtime, and Review. Require the
operator to confirm:

- `interleaved` topology;
- one model copy and one managed engine;
- intended memory nodes;
- whole-core CPU mask;
- canonical model and mount paths;
- full staging mode;
- staged, runtime, page-table, and operating-system reserve totals;
- zero blockers; and
- understood warnings.

Screenshot: Capacity or Review with the shared placement contract visible.

### 4. Prepare the workspace

Explain the token-bound confirmation, foreground sudo authorization, staged
copy progress, source-fingerprint verification, namespace construction, and
placement validation. Explain that cancellation is cooperative and that a
cancelled or failed Prepare rolls back mounts created by the operation.

Screenshots: Prepare confirmation and ready state.

### 5. Start and verify the engine

Start from Operate and verify the endpoint, managed process identity, reviewed
CPU mask, verified mounts, and source fingerprint. Show how to request a deep
refresh.

Screenshot: running state.

### 6. Stop without restaging

Stop the verified managed process group and confirm the state returns to ready
or stopped while the staged weights remain mounted.

Screenshot: stopped/ready state.

### 7. Destroy safely

Review the exact manifest and mount identities, confirm Destroy, and verify the
workspace becomes absent. State explicitly that the volatile weight mounts and
manifest are removed while durable usage, KV, and benchmark history remain.

Screenshot: final absent state.

### 8. Troubleshooting

Cover only problems on the core path:

- Prepare disabled by blockers;
- sudo ticket cannot be reused noninteractively;
- CPU or NUMA masks changed;
- a cancellable operation is still rolling back; and
- Stop is required before Destroy.

Link to `docs/ramdisk-tui.md` for the complete reference and additional modes.

### 9. Completion checklist

The reader verifies:

- no managed engine remains;
- no managed RAM-workspace mount remains;
- lifecycle status is absent;
- the canonical model remains unchanged; and
- expected durable state remains available.

## Screenshot architecture

### Capture source

The capture utility will instantiate the production
`ramdisk_textual.RamdiskTextualApp` with its supported dependency-injection
seams:

- `initial_snapshot` supplies a deterministic `ConsoleSnapshot`;
- `auto_refresh=False` prevents host discovery and background refresh;
- a non-authorizing `privilege_authorizer` prevents privileged work; and
- a documentation lifecycle double refuses every mutation if it is called.

The utility will use Textual's `App.run_test()` and pilot input to navigate the
real widgets. It will export SVG through `App.export_screenshot()` or
`App.save_screenshot()`.

### Fixture ownership

Documentation snapshots must not import from `c/tests/`. A small
documentation-specific fixture in the capture utility will define:

- one shared/interleaved plan;
- blocker-free illustrative memory and NUMA data;
- absent, ready, running, and stopped lifecycle reports;
- verified mount and namespace rows;
- a managed process row for the running state; and
- stable timestamps, ports, paths, and identifiers.

Keeping the fixture next to the capture code makes its illustrative nature
explicit and prevents production code from depending on tests.

### Viewport and assets

Captures use a fixed virtual terminal large enough to show the contract and
active step without clipping. The initial target is 110 columns by 34 rows;
the capture test will fail if the app's minimum-size guard is visible.

Assets will be written under:

`docs/media/ramdisk-tui/`

Expected assets:

- `01-inspect.svg`
- `02-review.svg`
- `03-prepare-confirmation.svg`
- `04-ready.svg`
- `05-running.svg`
- `06-stopped.svg`
- `07-absent.svg`

Every image receives descriptive alt text and a caption stating that hardware,
capacity, paths, identifiers, and endpoints are deterministic examples.

## Safety and failure behavior

The capture path must be safe by construction:

- no real model path is opened;
- no lifecycle mutation is permitted;
- no subprocess, engine, mount, sudo, or network operation is invoked;
- no automatic refresh reaches host discovery;
- the capture command exits nonzero if any expected step, review, or lifecycle
  state cannot be rendered;
- the lifecycle double raises immediately if a mutation unexpectedly crosses
  the UI boundary; and
- output is limited to the named documentation asset directory.

The guide must never suggest bypassing memory, cgroup, placement, mount-root,
symlink, swap, or sudo-policy blockers.

## Playwright validation

Playwright is an authoring-time documentation validator, not the TUI capture
backend and not an application dependency.

Validation will:

1. render the Markdown how-to into a temporary local HTML preview using the
   already available Markdown tooling;
2. serve the preview only on `127.0.0.1`;
3. launch Playwright against the installed system Chrome;
4. verify that every expected SVG request succeeds and every image has
   nonzero rendered dimensions;
5. verify local documentation links and anchors;
6. check that the main document, tables, images, and code blocks do not create
   unintended horizontal overflow at a desktop and a narrower viewport; and
7. take an authoring-only full-page preview screenshot for visual inspection.

The preview server, temporary HTML, and Playwright screenshot are not committed.
The generated TUI SVG assets are committed.

## Repository changes

Implementation is expected to add or update:

- `docs/ramdisk-tui-howto.md`;
- `docs/media/ramdisk-tui/*.svg`;
- one documentation capture utility under `c/tools/`;
- focused tests for safe deterministic capture;
- links in `README.md` and `docs/ramdisk-tui.md`; and
- documentation/developer instructions for regenerating the screenshots.

No production lifecycle module or public schema should change.

## Validation plan

1. Run the screenshot generator in a clean tracked worktree.
2. Confirm all expected SVGs exist, are nonempty, and contain no minimum-size
   warning.
3. Run the existing Textual pilot suite with Textual installed.
4. Run focused tests for the capture utility's no-mutation contract.
5. Render and inspect the guide with Playwright at desktop and narrow
   viewports.
6. Verify all added relative links and image paths.
7. Run `git diff --check`.
8. Run the repository's model-free check gate if capture-code changes can
   affect packaging or test discovery.

## Acceptance criteria

- A Linux operator can follow the guide from launch through verified Destroy
  without needing the reference guide for core-path instructions.
- The guide covers only shared, full-mode operation.
- Every screenshot comes from the real Textual app using deterministic data.
- Screenshot generation cannot mount, authorize sudo, launch an engine, or
  access a model.
- Screenshot captions distinguish illustrative data from operator-reviewed
  values.
- Playwright reports no broken images, broken local links, or unintended
  horizontal overflow in the preview.
- The existing TUI pilot tests and relevant model-free repository tests pass.
- README and reference-guide links make the new how-to discoverable.
