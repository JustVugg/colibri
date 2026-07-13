# Contributing

Keep changes focused and preserve Colibri's dependency-free default CPU path.

## Branches

- **`main`** is the stable branch. It's what users clone, and it stays known-good
  (engine always passes the token-exact oracle: `SNAP=./glm_tiny TF=1 ./glm 64 16 16`).
- **`dev`** is the integration branch. **Open your PR against `dev`.** Reviewed PRs
  land there first; once a batch is tested and stable, the maintainer fast-forwards
  it into `main`. This keeps `main` clean instead of taking every PR one at a time.

Every PR — on either branch — is reviewed for a clean build (0 warnings), the oracle
(32/32 TF + 20/20 greedy), and its own targeted validation before merge.

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
