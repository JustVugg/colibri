# NUMA Sampling Verifier Repair

Date: 2026-07-27

## Problem

RAM-disk preparation validates an interleaved tmpfs by mapping a bounded
selection of pages from each staged shard and reading the resulting node counts
from `/proc/self/numa_maps`. The current page selector walks each file with one
fixed stride. It adjusts that stride only until it is coprime with the number of
NUMA nodes.

That adjustment is sufficient when Linux allocates independent 4 KiB pages:
the selected page offsets rotate through every node residue. It is not
sufficient when tmpfs uses a larger folio. Linux selects the interleave node
from the folio index, effectively shifting the file-page index by the folio
order first. A stride that rotates at order 0 can become a multiple of the node
count after that shift.

For example, sampling 256 pages from a 1,048,576-page shard starts with a stride
of 4096. The current code advances it to 4097. With 2 MiB transparent huge
pages, the relevant index is shifted by nine, and the sampled huge-page index
advances by eight for the first 256 samples. On a two- or four-node host, those
samples can therefore report only one node even when the file is correctly
interleaved.

The failure message compounds the problem: it omits the observed counts and
prints `15%%`, so an operator cannot distinguish a real placement failure from
a sampling artifact.

## Goals

- Keep the bounded post-staging check of actual page placement.
- Prevent deterministic resonance at 4 KiB, intermediate mTHP, and 2 MiB page
  allocation granularities.
- Preserve the existing 15% maximum-deviation contract for interleaved mounts.
- Make a genuine failure actionable by reporting the sampled allocation and
  measured deviation.
- Correct doubled percent signs in the NUMA validation messages.
- Add deterministic regression coverage without requiring a multi-node test
  host.

## Non-goals

- Do not change tmpfs mount policy, staging topology, or THP defaults.
- Do not weaken a placement failure to a warning.
- Do not increase the existing sampling budget or map entire model shards.
- Do not add a dependency on `numactl`, privileged pagemap access, or
  `move_pages(2)` for interleaved validation.
- Do not refactor unrelated RAM-disk lifecycle code.

## Design

### Page selection

Replace the fixed-stride walk in `_sample_page_indices` with deterministic
stratified sampling.

The file is divided into `sample_pages` non-overlapping, nearly equal strata.
One page is selected from each stratum using a small fixed 64-bit mixing
function. Non-overlapping strata guarantee unique indices and retain coverage
across the file; mixed offsets prevent the low page-index bits from becoming
constant or periodic at a transparent-huge-page boundary.

Within each stratum, 32 fixed-salt offsets are evaluated. For each offset, the
selector models the Linux interleave residue at page orders 0 through 9:

```text
(page_index >> order) % node_count
```

The offset's score is the worst absolute distance from the running equal-share
target over those orders, followed by the sum of squared distances as a stable
secondary score. An order participates only when the file contains at least
seven allocation units per selected node at that order:

```text
ceil(total_pages / (1 << order)) >= 7 * node_count
```

Seven is `ceil(1 / 0.15)`, so one allocation unit cannot by itself exceed the
15% tolerance. The lowest-scoring offset is added to the sample and its residue
counts become the running state for the next stratum. The smallest salt is the
deterministic tie breaker.

This is a selection-only change. `_sample_numa_allocation` continues to touch
the chosen pages and obtains actual node counts from `/proc/self/numa_maps`.
The number of touched pages and the aggregate staging validation budget remain
unchanged.

The selector retains the current boundary behavior:

- Clamp the requested sample count to `[1, total_pages]`.
- Return every page when the entire file is sampled.
- Return unique, in-range integer page indices.
- Treat a single-node request as a coverage-only sample because no balance
  optimization is needed.

### Placement validation

Keep the current mask and balance checks:

1. Reject an empty allocation on a host where NUMA placement must be verified.
2. Reject more than 1% of sampled pages outside the reviewed memory-node mask.
3. For a node-local mount, require at least 95% on the target node.
4. For an interleaved multi-node mount, require every selected node to be
   within 15% of the ideal equal share.

When the interleaved check fails, report:

- the sorted `node=page_count` allocation;
- the largest measured deviation as a percentage; and
- the 15% permitted maximum.

For example:

```text
interleaved tmpfs sample is imbalanced: node pages 0=612, 1=388;
maximum deviation 22.4% exceeds 15%
```

Use literal single percent signs in this message and in the node-local 95%
message.

### Error and cleanup behavior

No lifecycle behavior changes. A failed check remains a preparation error, and
the existing preparation rollback continues to unmount the managed tmpfs and
persist the error manifest. The improved message supplies the evidence needed
to diagnose a real failure.

## Testing

Add focused unit tests for:

- the existing uniqueness, range, and full-sample behavior;
- two- and four-node residue balance at order 0 for a large power-of-two shard;
- two- and four-node residue balance at every eligible order through order 9,
  including the former `4097`-stride resonance;
- deterministic output for identical arguments;
- successful namespace validation with a balanced injected allocation;
- rejection of a genuinely imbalanced injected allocation;
- inclusion of node counts, measured deviation, and `15%` in the failure;
- a single `%` in the node-local `95%` failure.

Run the focused RAM-disk model-planning and mount test modules, followed by the
broader Python RAM-disk test suite used by the repository.

## Acceptance criteria

- The former power-of-two/THP sampling case is balanced for two and four nodes.
- `_sample_page_indices` remains deterministic, bounded, unique, and in range.
- A synthetic allocation outside the 15% limit still aborts preparation.
- Failure text includes actual counts and contains no doubled percent signs.
- Focused and broader RAM-disk tests pass.
