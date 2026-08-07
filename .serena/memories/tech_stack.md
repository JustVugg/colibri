# Technology stack
- Engine: portable C built with GCC/Clang/MinGW; OpenMP when available. Optional CUDA (C++17/NVCC), HIP, and Metal backends.
- Control plane/tests: Python >=3.10, primarily standard library and unittest. Guided RAM-disk TUI pins Textual >=8.2.8,<9.
- Web UI: TypeScript/Vite under `web/`; desktop shell: Rust/Tauri v2 under `desktop/`.
- Packaging: setuptools via `pyproject.toml`; Nix flake and GitHub Actions cover platform builds.
- Supported check matrix is Linux, native MSYS2/UCRT64 Windows, and macOS/clang.