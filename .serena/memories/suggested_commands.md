# Suggested commands
- Build native engine: `make` from repository root or `make -C c colibri`.
- Full dependency-light gate: `make check` or `make -C c check`.
- Python tests only: `make -C c test-python`.
- C tests only: `make -C c test-c`.
- Both suites without the clean/rebuild wrapper: `make -C c test`.
- Clean generated native artifacts: `make clean`.
- Run a focused Python test: from `c/`, `python3 -m unittest tests.test_<module>.<Class>.<test>`.
- RAM-disk lifecycle is Linux-only; scriptable entry point: `python3 c/coli ramdisk <action>`.