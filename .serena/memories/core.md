# Colibri project map
- Native engine and primary build/test system: `c/`; engine entrypoint `c/colibri.c`, small focused headers, Python launcher/control plane, tools, and dependency-light tests.
- Python package launcher: `colibri/`; packaging metadata in `pyproject.toml`.
- Browser client: `web/`; Tauri wrapper: `desktop/`; documentation: `docs/`; static site: `site/`.
- Repository-root `Makefile` delegates standard build/check/clean work to `c/Makefile`.
- Runtime design favors a flat, dependency-light C engine plus standard-library Python control-plane code.
- Read `mem:tech_stack` for build/runtime technologies, `mem:conventions` for local patterns, `mem:suggested_commands` for entry points, and `mem:task_completion` for verification gates.