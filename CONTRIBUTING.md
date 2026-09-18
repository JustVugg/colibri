# Contributing

Keep changes focused and preserve Colibri's dependency-free default CPU path.

## Branches

- **`main`** is the stable branch. It's what users clone, and it stays known-good
  (engine always passes the oracle: `SNAP=./glm_tiny TF=1 COLI_TEMP=0 ORACLE_STRICT=1 ORACLE_TF_MAX_MISMATCHES=2 ./colibri 64 16 16`,
  run from `c/`). The `glm_tiny` fixture is generated, not committed --
  `python3 c/tools/make_glm_oracle.py` builds it (needs torch).
- **`dev`** is the integration branch. **Open your PR against `dev`.** Reviewed PRs
  land there first; once a batch is tested and stable, the maintainer fast-forwards
  it into `main`. This keeps `main` clean instead of taking every PR one at a time.

Every PR — on either branch — is reviewed for a clean build (0 warnings), the oracle
(30–32/32 TF depending on floating-point near ties + 20/20 greedy), and its own
targeted validation before merge.

After generating the fixture, run both enforced comparisons from `c/`:

```sh
SNAP=./glm_tiny TF=1 COLI_TEMP=0 ORACLE_STRICT=1 ORACLE_TF_MAX_MISMATCHES=2 ./colibri 64 16 16
SNAP=./glm_tiny COLI_TEMP=0 ORACLE_STRICT=1 ./colibri 64 16 16
python3 tests/test_glm_oracle.py
```

`ORACLE_STRICT=1` makes a failed comparison exit with status 1. Strict mode is
token-exact by default. The teacher-forcing command above explicitly allows at
most two mismatches, preserving the 30–32/32 acceptance range for this fixture;
greedy comparison always requires every continuation token to match. Non-finite
output and incomplete generation fail regardless of the mismatch allowance.
Invalid reference arrays/JSON fail in either mode.
Without strict mode (or with `ORACLE_STRICT=0`), a completed comparison remains
report-only for diagnostic/benchmark callers; its exit status is not a correctness
gate. Strict mode rejects `REPLAY`, serving, text generation, and other execution
modes that would bypass the comparison.

`ORACLE_TF_MAX_MISMATCHES` is a nonnegative integer smaller than the number of
TF positions; it is used only for strict teacher-forcing comparisons. Unset or
`0` requires exact TF agreement. Generate the reference on the test host and
inspect near ties with `TF=1 DEBUG_LOGITS=1`; an allowed mismatch is still printed
and does not establish token-exact agreement. The regression script checks the
two-mismatch boundary, optional exact mode, and rejection of a single greedy
mismatch. It validates the original TF fixture within the same allowance. Routine
`make check` covers reference validation without needing torch or a generated model.

## Local checks

Run the lightweight checks locally:

```sh
make check
```

`make -C c check` remains available for scripts that already run from the
engine directory.

This performs one portable CPU build, C unit tests, and Python standard-library
tests. It does not download a model or require CUDA.

CUDA changes should additionally be checked on a CUDA-capable Linux host:

```sh
make -C c cuda-test CUDA_ARCH=native
```

Benchmark reports should include the commit, exact commands, hardware and
storage details, warm-up policy, run count, and median throughput.

Performance PRs should attach an experiment manifest based on
`docs/experiments/manifest.example.json`. Validate it before submission with:

```sh
python3 c/experiment_manifest.py path/to/result.json
```

The validator requires a full commit identity, at least three raw throughput
samples per arm, medians derived from those samples, hashed raw evidence, a
passing correctness gate, and exactly one changed configuration variable.
Negative and no-change results use the same record and remain first-class
evidence.
