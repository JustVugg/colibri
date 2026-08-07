# Task completion
- Run the smallest focused tests for changed behavior first.
- For C/Python engine or control-plane changes, finish with `make check` (equivalent engine-dir command: `make -C c check`).
- Changes touching platform-dependent code must account for the GitHub Actions Linux, MSYS2/UCRT64 Windows, and macOS matrix; local Linux success alone is insufficient.
- Run `git diff --check` before handoff.
- Report opt-in, privileged, model-dependent, GPU-dependent, or unavailable platform verification separately rather than implying it ran.