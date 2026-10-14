#!/usr/bin/env python3
"""colibri as a Model Context Protocol server, over stdio.

An AI coding assistant starts this process and talks JSON-RPC 2.0 to it, one
JSON message per line on stdin and stdout (the MCP stdio transport). Nothing
else may reach stdout: diagnostics go to stderr.

Tools, all thin wrappers over setup_flow.py (the same code `coli setup` runs):

    detect_hardware   RAM, disk, CPU features, GPUs (Vulkan, NVIDIA)
    recommend_models  the catalog against this machine, with the default marked
    install           build or fetch the engine and download a model, as a
                      background job that survives this server; progress is
                      reported with notifications/progress while waiting
    start / stop      the configured server (background)
    status            setup, install progress, server state, URLs, tok/s
    logs              the tail of the server, install or build log

Run it with `coli mcp`, or `python3 c/mcp_server.py`. docs/MCP_SERVER.md has
the client configuration.
"""
import json
import os
import subprocess
import sys
import threading
import time
import traceback

HERE = os.path.dirname(os.path.realpath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import setup_catalog  # noqa: E402
import setup_flow  # noqa: E402
import setup_hw  # noqa: E402

SUPPORTED_PROTOCOLS = ("2025-06-18", "2025-03-26", "2024-11-05")
SERVER_NAME = "colibri"

PARSE_ERROR, INVALID_REQUEST, METHOD_NOT_FOUND, INVALID_PARAMS, INTERNAL_ERROR = (
    -32700, -32600, -32601, -32602, -32603)

INSTRUCTIONS = (
    "colibri runs large mixture-of-experts language models locally. Typical flow: "
    "detect_hardware, recommend_models, install (a background job: follow it with status), "
    "start, then give the user the browser URL from the result. stop shuts the server down. "
    "Downloads are tens to hundreds of GB: confirm the model with the user before install.")


def _version():
    try:
        from version import __version__
        return __version__
    except ImportError:
        return "0"


def _schema(properties=None, required=()):
    return {"type": "object", "properties": properties or {}, "required": list(required),
            "additionalProperties": False}


TOOLS = [
    {"name": "detect_hardware", "title": "Detect hardware",
     "description": ("RAM (total and free), free disk where models go, CPU name and features, "
                     "and GPUs: Vulkan devices with type and memory, NVIDIA cards via nvidia-smi."),
     "inputSchema": _schema({"target_dir": {"type": "string",
                                            "description": "folder whose free space to report "
                                                           "(default ~/colibri-models)"}})},
    {"name": "recommend_models", "title": "Recommend models",
     "description": ("Every downloadable model against this machine: download size, RAM needed, "
                     "whether it fits and why, with the recommended default marked."),
     "inputSchema": _schema({"target_dir": {"type": "string"},
                             "include_unfit": {"type": "boolean", "default": False}})},
    {"name": "install", "title": "Install a model",
     "description": ("Build or fetch the engine (Vulkan/CUDA when a GPU is usable), download the "
                     "model with resume, write the run configuration, and by default start the "
                     "server in the background. Runs as a background job; returns at once unless "
                     "wait_seconds is set. Calling it again resumes an interrupted install."),
     "inputSchema": _schema({
         "model": {"type": "string", "description": "catalog id from recommend_models "
                                                    "(default: the recommendation)"},
         "target_dir": {"type": "string", "description": "folder for models"},
         "model_dir": {"type": "string", "description": "use a model already on disk"},
         "backend": {"type": "string", "enum": ["auto", "cpu", "vulkan", "cuda"]},
         "port": {"type": "integer", "minimum": 1, "maximum": 65535},
         "start": {"type": "boolean", "default": True},
         "wait_seconds": {"type": "integer", "minimum": 0, "maximum": 86400, "default": 0}})},
    {"name": "start", "title": "Start colibri",
     "description": "Start the configured server in the background; returns its URLs.",
     "inputSchema": _schema({"open_browser": {"type": "boolean", "default": False},
                             "wait_seconds": {"type": "integer", "minimum": 0, "maximum": 7200,
                                              "default": 0}})},
    {"name": "stop", "title": "Stop colibri",
     "description": "Stop the configured server and its engine.",
     "inputSchema": _schema()},
    {"name": "status", "title": "Status",
     "description": ("Configured model and engine, install progress, server state (stopped, "
                     "loading, ready), browser/OpenAI/Anthropic URLs, tok/s of the last answer."),
     "inputSchema": _schema()},
    {"name": "logs", "title": "Logs",
     "description": "The last lines of the server, install or build log.",
     "inputSchema": _schema({"which": {"type": "string", "enum": ["server", "install", "build"],
                                       "default": "server"},
                             "lines": {"type": "integer", "minimum": 1, "maximum": 5000,
                                       "default": 100}})},
]


class ToolError(Exception):
    """A tool-level failure: reported as a result with isError, not a protocol error."""


def _log(*parts):
    print("[colibri-mcp]", *parts, file=sys.stderr, flush=True)


def _check_args(name, args):
    tool = next((t for t in TOOLS if t["name"] == name), None)
    if tool is None:
        raise KeyError(name)
    if not isinstance(args, dict):
        raise ValueError("arguments must be an object")
    props = tool["inputSchema"]["properties"]
    for key, value in args.items():
        if key not in props:
            raise ValueError(f"unknown argument {key!r} for {name}")
        kind = props[key].get("type")
        ok = {"string": isinstance(value, str),
              "integer": isinstance(value, int) and not isinstance(value, bool),
              "boolean": isinstance(value, bool)}.get(kind, True)
        if not ok:
            raise ValueError(f"{name}.{key} must be a {kind}")
        if "enum" in props[key] and value not in props[key]["enum"]:
            raise ValueError(f"{name}.{key} must be one of {props[key]['enum']}")
        if kind == "integer":
            low, high = props[key].get("minimum"), props[key].get("maximum")
            if (low is not None and value < low) or (high is not None and value > high):
                raise ValueError(f"{name}.{key} is out of range")
    return args


# ---------------------------------------------------------------- tools


def tool_detect_hardware(args, notify):
    target = args.get("target_dir") or setup_flow.default_models_root()
    report = setup_hw.detect(target)
    report["summary"] = setup_hw.format_report(report)
    return report


def tool_recommend_models(args, notify):
    target = args.get("target_dir") or setup_flow.default_models_root()
    hw = setup_hw.detect(target, probe_gpu=False)
    rows = setup_flow._recommend_for(hw, args.get("target_dir"))
    models = [setup_catalog.as_dict(r) for r in rows if r["fits"] or args.get("include_unfit")]
    default = next((m["id"] for m in models if m["recommended"]), None)
    return {"recommended": default, "fits_means": setup_catalog.FITS_EXPLAINED,
            "ram_total_gb": round((hw["memory"].get("total") or 0) / 1e9, 1),
            "disk_free_gb": round((setup_hw.disk_free(target) or 0) / 1e9, 1),
            "target_dir": target, "models": models}


def _install_running():
    state = setup_flow.read_state() or {}
    if state.get("phase") in ("ready", "error", "interrupted", "done", "started", None):
        return None
    return state if setup_flow._pid_alive(state.get("pid")) else None


def _coli():
    return os.path.join(HERE, "coli")


_INSTALL_LOCK = threading.Lock()


def _launch_install(args):
    running = _install_running()
    if running:
        raise ToolError(f"an install is already running (pid {running.get('pid')}, phase "
                        f"{running.get('phase')}); follow it with status")
    if args.get("model"):
        try:
            setup_catalog.by_id(args["model"])
        except KeyError:
            raise ToolError(f"unknown model {args['model']!r}; recommend_models lists the ids")
    cmd = [sys.executable, _coli(), "setup", "--yes", "--no-browser"]
    if args.get("model"):
        cmd += ["--model", args["model"]]
    if args.get("model_dir"):
        cmd += ["--model-dir", args["model_dir"]]
    if args.get("target_dir"):
        cmd += ["--dir", args["target_dir"]]
    if args.get("backend"):
        cmd += ["--backend", args["backend"]]
    if args.get("port"):
        cmd += ["--port", str(args["port"])]
    cmd += ["--background"] if args.get("start", True) else ["--no-start"]
    log = setup_flow.log_path("install")
    os.makedirs(os.path.dirname(log), exist_ok=True)
    handle = open(log, "a", encoding="utf-8", errors="replace")
    handle.write(f"\n--- install {time.strftime('%Y-%m-%d %H:%M:%S')}: {' '.join(cmd)}\n")
    handle.flush()
    kwargs = {"stdin": subprocess.DEVNULL, "stdout": handle, "stderr": subprocess.STDOUT,
              "env": dict(os.environ, PYTHONUNBUFFERED="1")}
    if sys.platform == "win32":
        kwargs["creationflags"] = 0x00000200 | 0x08000000   # new group, hidden console
    else:
        kwargs["start_new_session"] = True   # survives the assistant closing this server
    process = setup_flow.spawn_detached(cmd, **kwargs)
    handle.close()
    # The job's own pid: the child rewrites this file as it goes, with the same pid.
    setup_flow._write_json(setup_flow.state_path(),
                           {"phase": "starting", "pid": process.pid, "updated": time.time()})
    return process, log, cmd


def tool_install(args, notify):
    # tools/call runs in independent threads: admission and publishing the
    # spawned job's state must be one operation, or two calls can both see no
    # running install. Waiting/progress is outside the lock so status remains
    # responsive and a second install receives the existing-job error promptly.
    with _INSTALL_LOCK:
        process, log, cmd = _launch_install(args)
    wait = int(args.get("wait_seconds") or 0)
    result = {"job": {"pid": process.pid, "log": log, "command": cmd}}
    deadline = time.time() + wait
    while wait and time.time() < deadline:
        code = process.poll()
        state = setup_flow.read_state() or {}
        if state.get("phase") == "download" and state.get("total"):
            notify(state.get("done", 0), state["total"],
                   f"downloading {state.get('model')}: {state.get('done', 0) / 1e9:.1f} of "
                   f"{state['total'] / 1e9:.1f} GB")
        elif state.get("phase"):
            notify(None, None, f"install: {state['phase']}")
        if code is not None:
            break
        time.sleep(1.0)
    result["exit_code"] = process.poll()
    result["install"] = setup_flow.read_state()
    if result["exit_code"] not in (None, 0):
        result["log_tail"] = setup_flow.tail_file(log, 30)
        raise ToolError(json.dumps(result, indent=2))
    result["next"] = ("the install continues in the background: call status to follow it"
                      if result["exit_code"] is None else "installed: call status for the URLs")
    return result


def _stderr_out(*parts):
    _log(*parts)


def tool_start(args, notify):
    cfg = setup_flow.load_config()
    if not cfg or not setup_flow.config_ready(cfg):
        raise ToolError("nothing is installed yet (or the install has not finished): call install")
    status = setup_flow.start_server(cfg, background=True,
                                     open_browser=bool(args.get("open_browser")), out=_stderr_out)
    wait = int(args.get("wait_seconds") or 0)
    deadline = time.time() + wait
    while wait and time.time() < deadline and status.get("state") != "ready":
        time.sleep(2.0)
        status = setup_flow.server_status(setup_flow.load_config() or cfg)
        notify(None, None, f"server: {status['state']}")
        if status["state"] == "stopped":
            status["log_tail"] = setup_flow.tail_file(setup_flow.log_path("serve"), 30)
            raise ToolError(json.dumps(status, indent=2))
    return status


def tool_stop(args, notify):
    cfg = setup_flow.load_config()
    if not cfg:
        raise ToolError("nothing is configured; there is nothing to stop")
    return setup_flow.stop_server(cfg, out=_stderr_out)


def tool_status(args, notify):
    return setup_flow.status_report()


def tool_logs(args, notify):
    which = {"server": "serve"}.get(args.get("which", "server"), args.get("which", "server"))
    path = setup_flow.log_path(which)
    return {"log": path, "text": setup_flow.tail_file(path, int(args.get("lines") or 100))}


HANDLERS = {"detect_hardware": tool_detect_hardware, "recommend_models": tool_recommend_models,
            "install": tool_install, "start": tool_start, "stop": tool_stop,
            "status": tool_status, "logs": tool_logs}


# ---------------------------------------------------------------- protocol


class Server:
    def __init__(self, out=None):
        self.out = out or sys.stdout
        self.lock = threading.Lock()
        self.initialized = False
        self.threads = []

    def send(self, message):
        line = json.dumps(message, separators=(",", ":"), ensure_ascii=True)
        with self.lock:
            self.out.write(line + "\n")
            self.out.flush()

    def error(self, msg_id, code, message):
        self.send({"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}})

    def result(self, msg_id, result):
        self.send({"jsonrpc": "2.0", "id": msg_id, "result": result})

    def handle_line(self, line):
        line = line.strip()
        if not line:
            return
        try:
            message = json.loads(line)
        except ValueError:
            self.error(None, PARSE_ERROR, "parse error")
            return
        if isinstance(message, list):
            for item in message:
                self.handle(item)
            return
        self.handle(message)

    def handle(self, message):
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
            self.error(message.get("id") if isinstance(message, dict) else None,
                       INVALID_REQUEST, "invalid request")
            return
        method = message.get("method")
        msg_id = message.get("id")
        is_request = "id" in message
        if method is None:
            return                      # a response to something we never asked: ignore
        params = message.get("params") or {}
        if not is_request:
            return                      # notifications: initialized, cancelled, ...
        if method == "initialize":
            requested = params.get("protocolVersion") if isinstance(params, dict) else None
            version = requested if requested in SUPPORTED_PROTOCOLS else SUPPORTED_PROTOCOLS[0]
            self.initialized = True
            self.result(msg_id, {"protocolVersion": version,
                                 "capabilities": {"tools": {"listChanged": False}},
                                 "serverInfo": {"name": SERVER_NAME, "title": "colibri",
                                                "version": _version()},
                                 "instructions": INSTRUCTIONS})
        elif method == "ping":
            self.result(msg_id, {})
        elif method == "tools/list":
            self.result(msg_id, {"tools": TOOLS})
        elif method == "tools/call":
            # Keep EOF draining limited to calls still in flight, rather than
            # retaining every completed worker for the lifetime of the session.
            self.threads = [worker for worker in self.threads if worker.is_alive()]
            thread = threading.Thread(target=self.call_tool, args=(msg_id, params), daemon=True)
            self.threads.append(thread)
            thread.start()
        elif method in ("resources/list", "prompts/list"):
            self.result(msg_id, {method.split("/")[0]: []})
        else:
            self.error(msg_id, METHOD_NOT_FOUND, f"method not found: {method}")

    def call_tool(self, msg_id, params):
        if not isinstance(params, dict) or not isinstance(params.get("name"), str):
            self.error(msg_id, INVALID_PARAMS, "tools/call needs a tool name")
            return
        name = params["name"]
        token = (params.get("_meta") or {}).get("progressToken") if isinstance(params.get("_meta"), dict) else None
        last = {"value": 0}

        def notify(done, total, text):
            # MCP requires progress to increase with every notification: bytes
            # when a download reports them, else one step past the last value.
            if token is None:
                return
            value = done if done is not None and done > last["value"] else last["value"] + 1
            last["value"] = value
            payload = {"progressToken": token, "progress": value}
            if total and total >= value:
                payload["total"] = total
            if text:
                payload["message"] = text
            self.send({"jsonrpc": "2.0", "method": "notifications/progress", "params": payload})

        try:
            args = _check_args(name, params.get("arguments", {}))
        except KeyError:
            self.error(msg_id, INVALID_PARAMS, f"unknown tool: {name}")
            return
        except ValueError as error:
            self.error(msg_id, INVALID_PARAMS, str(error))
            return
        try:
            data = HANDLERS[name](args, notify)
            self.result(msg_id, {"content": [{"type": "text", "text": json.dumps(data, indent=2)}],
                                 "structuredContent": data if isinstance(data, dict) else {"result": data},
                                 "isError": False})
        except (ToolError, setup_flow.SetupError) as error:
            self.result(msg_id, {"content": [{"type": "text", "text": str(error)}], "isError": True})
        except Exception as error:   # a tool bug must not take the server down
            _log(traceback.format_exc())
            self.result(msg_id, {"content": [{"type": "text",
                                              "text": f"{type(error).__name__}: {error}"}],
                                 "isError": True})

    def serve(self, stream=None):
        stream = stream or sys.stdin
        for line in stream:
            self.handle_line(line)
        # stdin closed: let the calls in flight answer (a stop drains its engine)
        # before exiting, within the minute a client gives a server to go.
        deadline = time.time() + 60
        for thread in self.threads:
            thread.join(timeout=max(0.0, deadline - time.time()))


def main():
    if hasattr(sys.stdin, "reconfigure"):
        try:
            sys.stdin.reconfigure(encoding="utf-8")
            sys.stdout.reconfigure(encoding="utf-8", newline="\n")
        except (AttributeError, OSError, ValueError):
            pass
    _log(f"colibri MCP server {_version()} on stdio")
    Server().serve()
    return 0


if __name__ == "__main__":
    sys.exit(main())
