# Project conventions
- Keep the native runtime readable and dependency-light; prefer focused headers around `c/colibri.c` over introducing large framework layers.
- Python control-plane tests use standard-library `unittest` and extensive `unittest.mock`; avoid model downloads and privileged operations in default tests.
- Platform claims are enforced by `.github/workflows/check.yml` on Linux, MSYS2/UCRT64 Windows, and macOS.
- Linux-only features must reject unsupported platforms cleanly and keep unsupported-platform tests from invoking POSIX-only internals accidentally.
- Security-sensitive RAM-disk operations verify paths, mount/process identities, durability, and ownership before chmod, signaling, mount, or unmount actions.
- Preserve behavior when feature environment variables are unset; opt-in paths must not weaken default engine behavior.