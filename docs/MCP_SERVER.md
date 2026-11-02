# MCP server: install and run colibri from an AI assistant

colibri ships a [Model Context Protocol](https://modelcontextprotocol.io) server,
so an AI coding assistant can detect the hardware, recommend a model, install
it, and start, check and stop the server through tool calls instead of shell
commands. It is the same code `coli setup` runs (`c/setup_flow.py`), written
with the Python standard library only: nothing to install beyond Python 3.10.

The shell path for assistants without MCP is [AI_SETUP.md](AI_SETUP.md).

## Run it

The transport is stdio: the client starts the process and exchanges JSON-RPC
2.0 messages with it, one per line. Either form works:

```bash
python3 /path/to/colibri/c/coli mcp
python3 /path/to/colibri/c/mcp_server.py
```

After `make install` the launcher is on the PATH and `coli mcp` is enough.

## Client configuration

Most MCP clients take a JSON entry of this shape (the file and the top-level
key differ by client; see its documentation for where it goes). Use absolute
paths.

Linux, macOS, WSL:

```json
{
  "mcpServers": {
    "colibri": {
      "command": "python3",
      "args": ["/home/me/colibri/c/coli", "mcp"]
    }
  }
}
```

Windows:

```json
{
  "mcpServers": {
    "colibri": {
      "command": "py",
      "args": ["-3", "C:\\Users\\me\\colibri\\c\\coli", "mcp"]
    }
  }
}
```

Optional environment, in an `"env"` object next to `"args"`:

| Variable | Use |
|---|---|
| `COLI_SETUP_HOME` | where the run configuration, state and logs live (default: `~/.local/share/colibri`, `~/Library/Application Support/colibri`, `%LOCALAPPDATA%\colibri`) |
| `HF_TOKEN` | token for gated model repositories |
| `HF_ENDPOINT` | a mirror of the model hub |

A client that only takes a command line: `python3 /home/me/colibri/c/coli mcp`.

## Tools

| Tool | Arguments | Returns |
|---|---|---|
| `detect_hardware` | `target_dir` (optional) | RAM total and free, free disk in `target_dir` (default `~/colibri-models`) with the WSL `/mnt/<drive>` warning, CPU name and features, the Vulkan device the engine would use (type, memory budget), NVIDIA cards, and a `summary` text |
| `recommend_models` | `target_dir`, `include_unfit` | the models that fit, each with `download_gb`, `ram_min_gb`, `ram_good_gb`, `dense_gb`, `status`, `reason`, and the `recommended` id; `fits_means` explains what fitting means |
| `install` | `model`, `target_dir`, `model_dir`, `backend` (`auto`, `cpu`, `vulkan`, `cuda`), `port`, `start` (default true), `wait_seconds` (default 0) | starts a background `coli setup --yes` job and returns its pid and log; with `wait_seconds` it waits that long and reports progress |
| `start` | `open_browser` (default false), `wait_seconds` | starts the configured server in the background; returns its state and URLs |
| `stop` | none | stops the configured server and its engine |
| `status` | none | the configuration (model, engine, backend, environment), install progress, server state (`stopped`, `loading`, `ready`), the browser, OpenAI and Anthropic URLs, the API `model_id`, tok/s of the last answer |
| `logs` | `which` (`server`, `install`, `build`), `lines` | the tail of that log |

Results carry the data twice, as JSON text in `content` and as
`structuredContent`. A tool that cannot do what was asked returns a result
with `isError: true` and the reason (for example `start` before anything is
installed); malformed calls get a JSON-RPC error (`-32602` for an unknown tool
or a bad argument, `-32601` for an unknown method).

### The install job

Downloads take minutes to hours, longer than a tool call should block. `install`
therefore starts the setup as a separate process that keeps running if the
assistant or this server exits, and returns at once. Follow it with `status`:
`install.phase` is `build`, `download`, `ready`, `started`, or `error` with
`install.message`; during the download `install.done` and `install.total` are
bytes. A second `install` while one runs is refused. Calling `install` again
after an error or an interruption resumes: finished files are kept, partial
ones continue.

With `wait_seconds`, and a `progressToken` in the request's `_meta`, the call
sends `notifications/progress` while it waits (bytes during the download, a
step count before).

## A typical session

1. `detect_hardware`, then `recommend_models`.
2. Tell the user the recommended model, its download size and where it goes;
   ask before downloading.
3. `install` with the chosen `model`. Poll `status` until `install.phase` is
   `started` (or `ready` with `start: false`).
4. If `logs` with `which: "install"` shows a line `to use the GPU through ...,
   first run: <command>`, the GPU needs a package: show the command to the user,
   and after they install it call `install` again with the same `model`.
5. `status` until `server.state` is `ready`, then give the user
   `server.urls.browser`, and for other apps `server.urls.openai_base_url`,
   `server.urls.anthropic_base_url` and `server.model_id`.
6. `stop` when the user is done.

## Protocol details

- Protocol versions `2025-06-18`, `2025-03-26` and `2024-11-05`; the server
  answers with the client's version when it supports it, else the newest.
- Capabilities: `tools`. The tool list does not change at runtime.
- stdout carries protocol messages only; diagnostics go to stderr, prefixed
  `[colibri-mcp]`.
- Requests are handled concurrently: a long `install` or `start` wait does not
  block `status` or `ping`.
- When stdin closes, calls in flight get up to a minute to answer, then the
  process exits. Background installs and servers it started keep running.

## Tests

`c/tests/test_mcp_server.py` drives the protocol (initialize, tools/list,
tools/call, errors, progress) in process and through a child process over real
stdin and stdout. By hand:

```bash
printf '%s\n' \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"t","version":"0"}}}' \
  '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"status","arguments":{}}}' \
  | python3 c/coli mcp
```
