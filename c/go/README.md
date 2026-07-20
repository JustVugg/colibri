# `coli` — Go port of the launcher (issue #310)

A dependency-free Go port of the colibrì `coli` CLI. It reproduces the Python
launcher's argument parsing, environment setup, and engine process management.
The goal (issue [#310](https://github.com/JustVugg/colibri/issues/310)) is a
Python-free *runtime path*.

This is the **first, incremental step**: the CLI process management is ported to
Go. The OpenAI-compatible gateway and the offline tooling stay in Python and are
invoked as subprocesses — so the Python code remains authoritative and its test
suite keeps passing. Nothing under `c/*.py` is modified.

## What runs where

| Command | Implementation |
|---------|----------------|
| `build` | native Go — `make -C c glm` |
| `info` | native Go — reads `config.json`, `/proc/meminfo`, disk, engine status |
| `run` | native Go — sets `PROMPT` and execs the engine |
| `chat` | native Go — engine subprocess, byte protocol, streaming markdown REPL |
| `serve`, `web` | delegate — spawn `openai_server.py` (the gateway stays Python) |
| `plan`, `doctor` | delegate — drive `resource_plan.py` / `doctor.py` via `python3` |
| `bench`, `convert` | delegate — run the Python tools (torch / tokenizers) |

`build`, `info`, `run`, and `chat` are Python-free. The rest still call Python,
matching the issue's scope (the gateway and offline tooling remain Python).

## Build

```sh
cd c && make coli-go        # produces c/coli-go
# or:
cd c/go && go build -o ../coli-go ./...
```

No third-party dependencies (`go.mod` has zero requires): the standard library
only, to preserve Colibri's dependency-free default path. The binary locates the
engine (`glm`) and the Python support files the same way the Python `coli` does;
`COLI_ENGINE` overrides the engine path.

## Parity

The port mirrors the Python `coli` byte-for-byte where it matters:

- identical help text, `info` rows, and error strings (`tests/test_cli_output.py`);
- identical child environment from `env_for`, including the measured Windows
  I/O/OMP defaults (`tests/test_env_defaults.py`) and `--auto-tier`;
- identical `plan` / `doctor` output (they call the same `resource_plan.py` /
  `doctor.py`);
- the same engine byte protocol: `\x01\x01READY\x01\x01\n` /
  `\x01\x01END\x01\x01\n` sentinels, `STAT` lines, `\x02RESET`/`\x02MORE` control
  writes, and two-stage Ctrl-C handling in `chat`.

## Known differences

- On Windows, `OMP_NUM_THREADS` defaults to the logical CPU count
  (`runtime.NumCPU()`) rather than the physical-core count the Python
  `physical_cpu_count()` computes. It is a `setdefault` the user can override, and
  Windows is not the primary target of this port.
- The `chat` input box is drawn once per prompt rather than redrawn with cursor
  math as the Python TTY version does; behaviour is otherwise identical.
