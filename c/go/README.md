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

The port mirrors the Python `coli` where it matters, and the Go tests
(`go test ./...`, gated by `make test-go`) pin it:

- identical help text and `info` rows;
- identical child environment from `env_for` — `env_test.go` spawns the real
  Python `env_for` as an oracle and diffs the child-env map byte-for-byte,
  including the measured Windows I/O/OMP defaults;
- identical `plan` / `doctor` output (they call the same `resource_plan.py` /
  `doctor.py`); `drivers_test.go` guards against those modules' APIs drifting;
- the same engine byte protocol: `\x01\x01READY\x01\x01\n` /
  `\x01\x01END\x01\x01\n` sentinels, `STAT` lines, `\x02RESET`/`\x02MORE` control
  writes, and two-stage Ctrl-C handling in `chat`.

Error messages are the Go CLI's own contract (pinned by `args_test.go`), not a
byte-clone of Python argparse — e.g. `--policy`'s invalid-choice text is
Go-native. The Go binary is the launcher going forward, so it owns these.

## Known differences

- On Windows, coli relays Ctrl-C by delivering `CTRL_BREAK_EVENT` to the child's
  process group (POSIX `SIGINT`/`SIGTERM` cannot be sent to another process on
  Windows). Whether the child shuts down *gracefully* depends on its own
  console-control handler — the engine (`glm.c`) and the Python gateway
  (`openai_server.py`); absent that, Ctrl-Break's default action still stops it.
- The `chat` input box is drawn once per prompt rather than redrawn with cursor
  math as the Python TTY version does; behaviour is otherwise identical.
