# GPU-aware RAM-disk Presets Implementation Plan

Design:
`docs/superpowers/specs/2026-07-27-gpu-aware-ramdisk-presets-design.md`

Goal: replace first-run low-level TUI setup with one shared preset resolver,
make GPU-local one-copy staging the default, persist the reviewed accelerator
contract, and make managed launch reproduce it exactly.

## Task 1: Add deterministic GPU/NUMA discovery

Files:

- Modify `c/ramdisk_support/discovery.py`
- Modify `c/ramdisk.py`
- Modify `c/tests/ramdisk_test_support.py`
- Modify `c/tests/test_ramdisk_model_planning.py`
- Modify `c/tests/test_ramdisk_facade.py`

Steps:

1. Add an injectable NVIDIA discovery helper that queries index, name, PCI bus
   ID, total VRAM, and free VRAM.
2. Normalize PCI identifiers and resolve Linux
   `/sys/bus/pci/devices/<id>/numa_node`.
3. Record explicit discovery states for missing `nvidia-smi`, malformed output,
   `-1` locality, inaccessible nodes, and driver/query failures.
4. Add normalized `gpus` and `gpu_discovery` records to Linux and unsupported
   hardware reports.
5. Extend the shared hardware fixture with optional sparse GPU/NUMA records.
6. Test a four-node/two-GPU fixture, duplicate nodes, `-1` on a single-node
   host, effective-mask rejection, and query failures.
7. Keep the facade dependency seams patchable and verify the direct discovery
   module and facade return identical normalized data.

## Task 2: Implement one shared preset resolver

Files:

- Add `c/ramdisk_support/presets.py`
- Modify `c/ramdisk.py`
- Add `c/tests/test_ramdisk_presets.py`
- Modify `c/tests/test_ramdisk_facade.py`

Steps:

1. Define stable preset IDs and labels for:

   - `gpu-fastest`;
   - `single`;
   - `minimal`; and
   - `replicas`.

2. Implement a pure resolver that receives the argument namespace, discovered
   hardware, scanned model, and injected authoritative plan builder.
3. For `gpu-fastest`, select the unique effective GPU-local NUMA nodes and
   their complete physical cores, retain shared/interleaved topology, and
   create one managed accelerator draft using all detected GPUs.
4. Ask the authoritative planner for the full candidate first. If memory
   admission fails, search bounded candidate capacities for the largest safe
   profile-guided shard closure without duplicating reserve formulas.
5. If GPU locality/CUDA capability is unavailable, return the Single draft
   with a visible fallback explanation.
6. Make Minimal profile-guided only and return an actionable blocking result
   when the profile is missing or incompatible.
7. Make Replicas set `per-node` explicitly; Auto must never reach this branch.
8. Return populated argument values, preset metadata, and the resulting plan
   without mounting, copying, persisting, or launching anything.
9. Test exact draft values, one copy/one engine for two GPUs, full/partial
   selection, missing profile behavior, fallback explanations, and explicit
   replication.

## Task 3: Persist an authoritative accelerator contract

Files:

- Modify `c/ramdisk_support/planning.py`
- Modify `c/ramdisk_support/state.py`
- Modify `c/ramdisk_support/processes.py`
- Modify `c/ramdisk.py`
- Modify `c/tests/test_ramdisk_planning_module.py`
- Modify `c/tests/test_ramdisk_state_lifecycle.py`
- Modify `c/tests/test_ramdisk_processes.py`

Steps:

1. Normalize the resolver's internal accelerator draft into a JSON-safe
   `managed_accelerator` plan section.
2. Include mode, selected devices, PCI/NUMA evidence, mmap/RAM-map selection,
   async-copy setting, and VRAM-budget policy.
3. Add planner blockers for malformed devices, devices outside discovery, and
   accelerator nodes outside the reviewed effective memory mask.
4. Validate the persisted section before lifecycle code may consume it.
5. Revalidate detected device identity and effective NUMA visibility at start
   so device disappearance or namespace drift fails closed.
6. Preserve compatibility for existing manifests without an accelerator
   section by treating them as the current CPU/direct-RAM-map contract.
7. Cover JSON round trips, malformed state rejection, legacy defaults, and
   device/NUMA drift.

## Task 4: Apply the reviewed accelerator environment

Files:

- Add `c/ramdisk_support/accelerator.py`
- Modify `c/ramdisk_support/lifecycle.py`
- Modify `c/ramdisk_support/benchmark.py`
- Modify `c/ramdisk.py`
- Modify `c/tests/test_ramdisk_state_lifecycle.py`
- Modify `c/tests/test_ramdisk_benchmark_module.py`

Steps:

1. Centralize ambient accelerator-variable sanitization and managed
   environment generation.
2. Continue removing ambient `COLI_CUDA`, `COLI_GPU(S)`,
   `CUDA_EXPERT_GB`, `COLI_MMAP`, `COLI_RAMMAP`, and async-copy values.
3. For `gpu-fastest`, apply:

   ```text
   COLI_CUDA=1
   COLI_GPUS=<reviewed devices>
   CUDA_EXPERT_GB=auto
   COLI_CUDA_ASYNC=1
   COLI_MMAP=1
   COLI_RAMMAP=0
   ```

   Use `COLI_GPU` for a single reviewed device.

4. For a legacy or CPU plan, retain the current
   `COLI_RAMMAP=1`/`COLI_MMAP=0` behavior.
5. Use the same helper for managed start and benchmark construction.
6. Persist the exact applied accelerator environment in process/benchmark
   records for diagnosis.
7. Test hostile ambient overrides, single/multiple devices, mmap conflict
   prevention, legacy behavior, and benchmark parity.

## Task 5: Expose preset and GPU evidence in presentation

Files:

- Modify `c/ramdisk_support/presentation.py`
- Modify `c/ramdisk_ui.py`
- Modify `c/tests/test_ramdisk_presentation.py`
- Modify `c/tests/test_ramdisk_presentation_module.py`
- Modify `c/tests/test_ramdisk_ui.py`

Steps:

1. Add pure presentation rows for the first-run question and its four choices.
2. Show the selected preset or Custom state in settings/review.
3. Show GPU index/name, PCI bus, NUMA node, selected GPU-local nodes, one-copy
   engine count, mmap upload mode, and fallback reason.
4. Keep existing blockers, warnings, placement contracts, and confirmation
   identities authoritative.
5. Ensure Replicas is labeled as independent engines and multiplied memory,
   never as multi-GPU sharding.
6. Add compact and full rendering tests for GPU, fallback, Custom, and legacy
   plans.

## Task 6: Add the Textual first-run preset screen

Files:

- Modify `c/ramdisk_textual.py`
- Modify `c/tests/test_ramdisk_textual.py`
- Modify `c/tools/capture_ramdisk_tui.py`

Steps:

1. Add a keyboard-accessible modal shown only after the first coherent status
   refresh proves that no manifest exists.
2. Focus Fastest GPU staging by default and expose all four choices.
3. Resolve the choice through the lifecycle facade, replace only the draft
   arguments, refresh the plan, and jump to Review.
4. Never show the modal for prepared, running, stopped, or error manifests.
5. Mark the preset Custom after any preset-controlled advanced edit while
   retaining the accelerator draft unless that edit explicitly changes it.
6. Keep the modal truthful at the minimum supported viewport and preserve
   quit/termination behavior.
7. Test default keyboard selection, button selection, direct Review
   navigation, Custom edits, existing-manifest bypass, and refresh races.
8. Update deterministic TUI captures that now begin with the preset question.

## Task 7: Add the curses first-run preset screen

Files:

- Modify `c/ramdisk_support/curses_ui.py`
- Modify `c/tests/test_ramdisk_curses_ui_module.py`

Steps:

1. Add a first-run state that is entered only after status confirms no
   manifest.
2. Render the same four shared presentation choices and make Enter select
   Fastest GPU staging by default; support direct `1` through `4` selection.
3. Apply the shared resolver, invalidate cached planning state, and return to
   the plan/review screen.
4. Skip selection for any persisted manifest and preserve cancellation,
   resize, signal, and background-operation semantics.
5. Mark subsequent settings edits Custom.
6. Test default/direct selection, exact resolver arguments, plan-cache
   invalidation, existing-manifest bypass, and small-terminal behavior.

## Task 8: Document and verify the completed workflow

Files:

- Modify `docs/ramdisk-tui.md`
- Modify `docs/SETTINGS.md`
- Modify `README.md` if the top-level workflow text needs a one-line update
- Review all modified source, tests, captures, and documentation

Steps:

1. Document the first-run question, default GPU-aware behavior, fallback rules,
   advanced editing, and existing-manifest bypass.
2. State that one shared copy across two GPU-local nodes improves topology but
   cannot make every page local to both GPUs.
3. Explain why multiple tmpfs mounts are not RAID and why Replicas remains
   explicit.
4. Run focused tests:

   ```bash
   python -m unittest \
     c.tests.test_ramdisk_presets \
     c.tests.test_ramdisk_model_planning \
     c.tests.test_ramdisk_planning_module \
     c.tests.test_ramdisk_state_lifecycle \
     c.tests.test_ramdisk_benchmark_module \
     c.tests.test_ramdisk_presentation \
     c.tests.test_ramdisk_presentation_module \
     c.tests.test_ramdisk_textual \
     c.tests.test_ramdisk_curses_ui_module
   ```

5. Run the full repository verification:

   ```bash
   make -C c check
   ```

6. Run `git diff --check`, inspect the complete diff, and confirm `.serena/`
   and unrelated user files remain untouched.
7. Report test counts, environment-gated skips, the final commit, and the
   remaining hardware-validation requirement for the target four-node/two-GPU
   system.
