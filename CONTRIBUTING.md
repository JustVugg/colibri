# Contributing

Keep changes focused and preserve Colibri's dependency-free default CPU path.

## Local checks

Run the lightweight checks locally:

```sh
make -C c check
```

This performs one portable CPU build, C unit tests, and Python standard-library
tests. It does not download a model or require CUDA.

CUDA changes should additionally be checked on a CUDA-capable Linux host:

```sh
make -C c cuda-test CUDA_ARCH=native
```

Benchmark reports should include the commit, exact commands, hardware and
storage details, warm-up policy, run count, and median throughput.
