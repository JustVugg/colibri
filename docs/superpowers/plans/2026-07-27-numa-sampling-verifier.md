# NUMA Sampling Verifier Implementation Plan

Design: `docs/superpowers/specs/2026-07-27-numa-sampling-verifier-design.md`

Goal: prevent THP-granularity resonance in the bounded tmpfs NUMA sampler while
retaining hard failures for genuine placement imbalance and making those
failures actionable.

## Task 1: Lock down the sampling regression

Files:

- Modify `c/tests/test_ramdisk_model_planning.py`

Steps:

1. Extend the NUMA sampling test to exercise a 1,048,576-page shard with 256
   samples on two and four nodes.
2. For every page order from 0 through 9 that has at least seven allocation
   units per selected node, count:

   ```python
   (index >> order) % nodes
   ```

3. Assert that every node's count is within 15% of the ideal count.
4. Assert that repeated calls with identical arguments return identical
   indices and that indices are unique and in range.
5. Run:

   ```bash
   python -m unittest \
     c.tests.test_ramdisk_model_planning.RamdiskPlanningTest.test_numa_sampling_avoids_page_order_resonance
   ```

6. Confirm the existing fixed-stride implementation fails at the THP order.

## Task 2: Lock down validation diagnostics

Files:

- Modify `c/tests/test_ramdisk_mounts.py`

Steps:

1. Add a helper that stages the fixture namespace in a temporary directory so
   `_validate_namespace` can be tested with an injected sampler.
2. Add a balanced interleaved allocation case and assert validation succeeds.
3. Add an imbalanced interleaved allocation case and assert the exception
   contains:

   - sorted node page counts;
   - the measured maximum deviation;
   - `15%`, not `15%%`.

4. Add or extend the node-local failure case to assert `95%`, not `95%%`.
5. Run the new tests and confirm the diagnostic assertions fail before the
   production change.

## Task 3: Implement deterministic stratified selection

Files:

- Modify `c/ramdisk_support/mounts.py`

Steps:

1. Add a private 64-bit mixing helper using fixed integer constants and explicit
   masking so output is stable across Python versions and processes.
2. Preserve `_sample_page_indices` input clamping and the full-file fast path.
3. Divide the page range into `sample_pages` non-overlapping integer strata.
4. For each of 32 fixed salts, select one mixed offset from every stratum.
5. Score each candidate by its worst relative node-residue deviation across
   eligible page orders 0 through 9.
6. Return the lowest-scoring candidate; break equal scores by the lowest salt.
7. Keep the single-node path simple because it needs coverage, not balance
   optimization.
8. Re-run the regression from Task 1 and confirm it passes.

## Task 4: Improve validation errors

Files:

- Modify `c/ramdisk_support/mounts.py`

Steps:

1. Compute all selected-node deviations once and retain the maximum.
2. Format allocation counts in selected-node order as `node=count`.
3. Raise an error containing the counts, measured maximum deviation to one
   decimal place, and the 15% limit.
4. Change the node-local literal from `95%%` to `95%`.
5. Run the diagnostics tests from Task 2 and confirm they pass.

## Task 5: Verify scope and regressions

Files:

- Review all modified source, test, and plan files.

Steps:

1. Run:

   ```bash
   python -m unittest \
     c.tests.test_ramdisk_model_planning \
     c.tests.test_ramdisk_mounts
   ```

2. Discover the repository's broader RAM-disk Python test command from the
   Makefile or existing test instructions and run it.
3. Run `git diff --check`.
4. Inspect `git diff` and confirm no lifecycle, mount-policy, THP-default, or
   sampling-budget changes were introduced.
5. Report focused/broader test counts, any environment-gated skips, and the
   exact changed files.
