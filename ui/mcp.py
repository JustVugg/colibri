"""MCP connections for the chat: the tools of MCP servers, offered to the model like web search.

The servers are kept in one JSON file (--mcp-config, COLIBRI_MCP, else ~/colibri-mcp.json), in
the shape Bionic uses:

    {"servers": [{"id": "...", "name": "comfy-mcp", "enabled": true,
                  "connection": {"type": "stdio", "command": "comfy-mcp.exe", "args": [],
                                 "cwd": "E:\\ComfyUI", "env": {}, "timeoutMs": 1200000}},
                 {"id": "...", "name": "remote", "enabled": true,
                  "connection": {"type": "streamableHttp", "url": "https://.../mcp", "headers": {}}}]}

A server starts on first use (its tool list or a call) and keeps running; a stdio server that
exits is started again on the next use. Two transports: stdio (newline-delimited JSON-RPC on the
program's stdin/stdout) and streamable HTTP (POST, answered as JSON or as an SSE stream). OAuth
sign-in is not supported. The model sees each tool as <server>__<tool>. Standard library only."""
import itertools
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid

CONFIG = os.environ.get("COLIBRI_MCP") or os.path.join(os.path.expanduser("~"), "colibri-mcp.json")
PROTOCOL = "2025-06-18"
CLIENT = {"name": "colibri-ui", "version": "1.0"}
RESULT_MAX = 6000            # characters of one tool result given to the model (~1,500 tokens:
                             # a research turn reads many pages, and all of them stay in the context)
DEFAULT_TIMEOUT = 120.0      # seconds, when a server names none
_lock = threading.RLock()
_clients = {}                # server id -> client


# ------------------------------------------------------------------ config

def _load():
    try:
        with open(CONFIG, encoding="utf-8") as f:
            c = json.load(f)
        if isinstance(c, dict) and isinstance(c.get("servers"), list):
            return c
    except (OSError, ValueError):
        pass
    return {"servers": []}


def _store(c):
    d = os.path.dirname(os.path.abspath(CONFIG))
    os.makedirs(d, exist_ok=True)
    tmp = CONFIG + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(c, f, indent=2)
    os.replace(tmp, CONFIG)


def _slug(name):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", name or "").strip("_-")[:24] or "mcp"


def _normalize(s):
    """one server entry from what a person pastes: Bionic's shape, or the mcpServers shape
    ({"command", "args", "env"} or {"url", "headers"}) with a name"""
    if not isinstance(s, dict):
        raise ValueError("a server is a JSON object")
    conn = s.get("connection")
    if not isinstance(conn, dict):
        if s.get("url"):
            conn = {"type": "streamableHttp", "url": s["url"], "headers": s.get("headers") or {}}
        elif s.get("command"):
            conn = {"type": "stdio", "command": s["command"], "args": s.get("args") or [],
                    "env": s.get("env") or {}}
            if s.get("cwd"):
                conn["cwd"] = s["cwd"]
        else:
            raise ValueError("a server needs a command (stdio) or a url (HTTP)")
        if s.get("timeoutMs"):
            conn["timeoutMs"] = s["timeoutMs"]
    name = str(s.get("name") or "").strip()
    if not name:
        raise ValueError("a server needs a name")
    t = conn.get("type")
    if t == "stdio":
        if not isinstance(conn.get("command"), str) or not conn["command"].strip():
            raise ValueError("a stdio server needs a command")
        if not isinstance(conn.get("args", []), list) or not all(isinstance(a, str) for a in conn.get("args", [])):
            raise ValueError("args is a list of strings")
        if not isinstance(conn.get("env", {}), dict):
            raise ValueError("env is an object of strings")
    elif t in ("streamableHttp", "http", "sse"):
        conn["type"] = "streamableHttp"
        if not re.match(r"^https?://", str(conn.get("url") or "")):
            raise ValueError("an HTTP server needs an http(s) url")
        if (conn.get("auth") or {}).get("type") == "oauth":
            pass   # kept as written; the connection will say sign-in is not supported
    else:
        raise ValueError("connection type is stdio or streamableHttp")
    return {"id": str(s.get("id") or uuid.uuid4()), "name": name, "description": str(s.get("description") or ""),
            "enabled": bool(s.get("enabled", True)), "connection": conn}


# ------------------------------------------------------------------ clients

class McpError(Exception):
    pass


class _Base:
    def __init__(self, server):
        self.server = server
        self.conn = server["connection"]
        self.timeout = max(5.0, float(self.conn.get("timeoutMs") or DEFAULT_TIMEOUT * 1000) / 1000)
        self.ids = itertools.count(1)
        self.tools = None
        self.error = None
        self.info = {}
        self.lock = threading.Lock()

    def ensure(self):
        with self.lock:
            if self.tools is not None and self.alive():
                return
            self.close()
            self.tools = None
            try:
                self.open()
                r = self.request("initialize", {"protocolVersion": PROTOCOL, "capabilities": {},
                                                "clientInfo": CLIENT}, 60)
                self.info = r.get("serverInfo") or {}
                self.notify("notifications/initialized")
                tools, cursor = [], None
                for _ in range(20):
                    r = self.request("tools/list", {"cursor": cursor} if cursor else {}, 60)
                    tools += r.get("tools") or []
                    cursor = r.get("nextCursor")
                    if not cursor:
                        break
                self.tools = tools
                self.error = None
            except Exception as e:   # noqa: BLE001 - said on the page
                self.error = str(e) or type(e).__name__
                self.close()
                raise McpError(self.error)

    def call(self, tool, args):
        self.ensure()
        return self.request("tools/call", {"name": tool, "arguments": args or {}}, self.timeout)


class _Stdio(_Base):
    def __init__(self, server):
        super().__init__(server)
        self.proc = None
        self.pending = {}
        self.plock = threading.Lock()
        self.stderr = []

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def _argv(self):
        cmd = os.path.expandvars(os.path.expanduser(self.conn["command"]))
        exe = shutil.which(cmd) or cmd
        argv = [exe] + list(self.conn.get("args") or [])
        if os.name == "nt" and exe.lower().endswith((".cmd", ".bat")):
            argv = ["cmd.exe", "/d", "/s", "/c"] + argv     # npm's shims are batch files
        return argv

    def open(self):
        env = dict(os.environ)
        env.update({k: str(v) for k, v in (self.conn.get("env") or {}).items()})
        cwd = self.conn.get("cwd") or None
        if cwd and not os.path.isdir(cwd):
            raise McpError("working folder not found: %s" % cwd)
        flags = 0x08000000 if os.name == "nt" else 0       # CREATE_NO_WINDOW: no console pops up
        try:
            self.proc = subprocess.Popen(self._argv(), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE, cwd=cwd, env=env, creationflags=flags)
        except OSError as e:
            raise McpError("could not start %s: %s" % (self.conn["command"], e))
        self.stderr = []
        threading.Thread(target=self._read, args=(self.proc,), daemon=True).start()
        threading.Thread(target=self._read_err, args=(self.proc,), daemon=True).start()

    def _read_err(self, proc):
        for line in iter(proc.stderr.readline, b""):
            self.stderr.append(line.decode("utf-8", "replace").rstrip())
            del self.stderr[:-40]

    def _read(self, proc):
        for line in iter(proc.stdout.readline, b""):
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            if not isinstance(msg, dict):
                continue
            if "method" in msg and "id" in msg:            # a request from the server
                ans = {"jsonrpc": "2.0", "id": msg["id"]}
                if msg["method"] == "ping":
                    ans["result"] = {}
                elif msg["method"] == "roots/list":
                    ans["result"] = {"roots": []}
                else:
                    ans["error"] = {"code": -32601, "message": "not supported by colibri"}
                self._send(ans)
                continue
            q = None
            with self.plock:
                q = self.pending.pop(msg.get("id"), None)
            if q:
                q.put(msg)
        with self.plock:                                    # the program ended: wake every waiter
            for q in self.pending.values():
                q.put({"error": {"message": "the server exited"}})
            self.pending.clear()

    def _send(self, msg):
        try:
            self.proc.stdin.write((json.dumps(msg) + "\n").encode())
            self.proc.stdin.flush()
        except (OSError, AttributeError) as e:
            raise McpError("the server is not running (%s)" % e)

    def notify(self, method, params=None):
        self._send({"jsonrpc": "2.0", "method": method, **({"params": params} if params else {})})

    def request(self, method, params, timeout):
        rid = next(self.ids)
        q = queue.Queue()
        with self.plock:
            self.pending[rid] = q
        self._send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
        try:
            msg = q.get(timeout=timeout)
        except queue.Empty:
            with self.plock:
                self.pending.pop(rid, None)
            try:
                self.notify("notifications/cancelled", {"requestId": rid, "reason": "timeout"})
            except McpError:
                pass
            raise McpError("%s timed out after %d s" % (method, timeout))
        if "error" in msg:
            err = msg["error"] or {}
            tail = (" | " + " / ".join(self.stderr[-3:])) if self.stderr and not self.alive() else ""
            raise McpError("%s%s" % (err.get("message") or err, tail))
        return msg.get("result") or {}

    def close(self):
        p, self.proc = self.proc, None
        if p and p.poll() is None:
            try:
                p.stdin.close()
                p.wait(timeout=3)
            except Exception:   # noqa: BLE001
                pass
            if p.poll() is None:
                if os.name == "nt":   # the batch shim's children too
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)], capture_output=True)
                else:
                    p.kill()


class _Http(_Base):
    def __init__(self, server):
        super().__init__(server)
        self.session = None
        self.opened = False

    def alive(self):
        return self.opened

    def open(self):
        if (self.conn.get("auth") or {}).get("type") == "oauth":
            raise McpError("this server needs an OAuth sign-in, which colibri does not support yet")
        self.session = None
        self.opened = True

    def _post(self, msg, timeout):
        h = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream",
             "MCP-Protocol-Version": PROTOCOL}
        h.update({k: str(v) for k, v in (self.conn.get("headers") or {}).items()})
        if self.session:
            h["Mcp-Session-Id"] = self.session
        req = urllib.request.Request(self.conn["url"], data=json.dumps(msg).encode(), headers=h, method="POST")
        try:
            return urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as e:
            body = e.read(2000).decode("utf-8", "replace")
            if e.code == 401:
                raise McpError("the server wants a sign-in (401): put a token in headers")
            raise McpError("HTTP %d: %s" % (e.code, body[:300]))
        except OSError as e:
            raise McpError("cannot reach %s: %s" % (self.conn["url"], e))

    def notify(self, method, params=None):
        with self._post({"jsonrpc": "2.0", "method": method, **({"params": params} if params else {})}, 30):
            pass

    def request(self, method, params, timeout):
        rid = next(self.ids)
        r = self._post({"jsonrpc": "2.0", "id": rid, "method": method, "params": params}, timeout)
        with r:
            sid = r.headers.get("Mcp-Session-Id")
            if sid:
                self.session = sid
            ctype = r.headers.get("Content-Type", "")
            msg = None
            if "text/event-stream" in ctype:
                data = []
                for raw in r:
                    line = raw.decode("utf-8", "replace").rstrip("\r\n")
                    if line.startswith("data:"):
                        data.append(line[5:].lstrip())
                    elif not line and data:
                        try:
                            m = json.loads("\n".join(data))
                        except ValueError:
                            m = None
                        data = []
                        if isinstance(m, dict) and m.get("id") == rid and "method" not in m:
                            msg = m
                            break
            else:
                msg = json.loads(r.read() or b"{}")
        if not isinstance(msg, dict):
            raise McpError("%s: no answer from the server" % method)
        if "error" in msg:
            raise McpError(str((msg["error"] or {}).get("message") or msg["error"]))
        return msg.get("result") or {}

    def close(self):
        self.opened = False
        self.session = None


def _client(server):
    with _lock:
        c = _clients.get(server["id"])
        if c and c.conn == server["connection"]:
            return c
        if c:
            c.close()
        c = (_Stdio if server["connection"]["type"] == "stdio" else _Http)(server)
        _clients[server["id"]] = c
        return c


def _drop(sid):
    with _lock:
        c = _clients.pop(sid, None)
    if c:
        c.close()


def shutdown():
    for sid in list(_clients):
        _drop(sid)


# ------------------------------------------------------------------ what the page asks

def _view(s):
    c = _clients.get(s["id"])
    st = "off" if not s["enabled"] else ("ready" if c and c.tools is not None and c.alive() else
                                          "error" if c and c.error else "idle")
    conn = s["connection"]
    return {"id": s["id"], "name": s["name"], "description": s.get("description", ""), "enabled": s["enabled"],
            "type": conn["type"], "target": conn.get("url") or " ".join([conn.get("command", "")] + list(conn.get("args") or [])),
            "state": st, "error": c.error if c else None,
            "stderr": (c.stderr[-6:] if c and isinstance(c, _Stdio) else []),
            "tools": [t.get("name") for t in (c.tools or [])] if c else []}


def list_servers():
    return {"config": CONFIG, "servers": [_view(s) for s in _load()["servers"]]}


def get(sid):
    for s in _load()["servers"]:
        if s["id"] == sid:
            return s
    raise ValueError("no server %r" % sid)


def save(entry):
    s = _normalize(entry)
    c = _load()
    names = {x["name"].lower(): x["id"] for x in c["servers"]}
    if names.get(s["name"].lower(), s["id"]) != s["id"]:
        raise ValueError("there is already a server named %s" % s["name"])
    for i, x in enumerate(c["servers"]):
        if x["id"] == s["id"]:
            c["servers"][i] = s
            break
    else:
        c["servers"].append(s)
    _store(c)
    _drop(s["id"])
    return {"id": s["id"], "name": s["name"]}


def delete(sid):
    c = _load()
    c["servers"] = [x for x in c["servers"] if x["id"] != sid]
    _store(c)
    _drop(sid)
    return {"id": sid}


def enable(sid, on):
    c = _load()
    for x in c["servers"]:
        if x["id"] == sid:
            x["enabled"] = bool(on)
    _store(c)
    if not on:
        _drop(sid)
    return {"id": sid, "enabled": bool(on)}


def connect(sid):
    """start (or start again) one server and list its tools"""
    s = get(sid)
    _drop(sid)
    try:
        _client(s).ensure()
    except McpError:
        pass
    return _view(s)


def import_file(path):
    """servers from Bionic's ng-mcp.json ({"servers": [...]}) or an mcpServers file (Claude,
    LM Studio, Cursor: {"mcpServers": {name: {...}}}); a name already here is skipped"""
    path = os.path.expandvars(os.path.expanduser((path or "").strip().strip('"')))
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError) as e:
        raise ValueError("cannot read %s: %s" % (path, e))
    entries = []
    if isinstance(data.get("servers"), list):
        entries = data["servers"]
    elif isinstance(data.get("mcpServers"), dict):
        entries = [dict(v, name=k) for k, v in data["mcpServers"].items() if isinstance(v, dict)]
    else:
        raise ValueError("no servers in %s (expected \"servers\" or \"mcpServers\")" % path)
    have = {x["name"].lower() for x in _load()["servers"]}
    added, skipped = [], []
    for e in entries:
        try:
            s = _normalize(e)
        except ValueError as err:
            skipped.append({"name": str(e.get("name") if isinstance(e, dict) else "?"), "why": str(err)})
            continue
        if s["name"].lower() in have:
            skipped.append({"name": s["name"], "why": "already here"})
            continue
        s["id"] = str(uuid.uuid4())
        c = _load()
        c["servers"].append(s)
        _store(c)
        have.add(s["name"].lower())
        added.append(s["name"])
    return {"added": added, "skipped": skipped}


def _fname(server, tool):
    return ("%s__%s" % (_slug(server["name"]), re.sub(r"[^A-Za-z0-9_-]+", "_", tool)))[:64]


def tools():
    """the tools of every enabled server, as chat-completions tools, started as needed; a server
    that fails is listed in errors and left out"""
    out, errors = [], []
    servers = [s for s in _load()["servers"] if s["enabled"]]
    results = {}

    def one(s):
        try:
            c = _client(s)
            c.ensure()
            results[s["id"]] = c.tools
        except McpError as e:
            errors.append({"server": s["name"], "error": str(e)})

    ts = [threading.Thread(target=one, args=(s,)) for s in servers]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    for s in servers:
        for t in results.get(s["id"]) or []:
            schema = t.get("inputSchema") or {"type": "object", "properties": {}}
            desc = (t.get("description") or t.get("title") or "")[:1024]
            out.append({"type": "function", "function": {
                "name": _fname(s, t["name"]),
                "description": "[%s] %s" % (s["name"], desc) if desc else "[%s]" % s["name"],
                "parameters": schema}})
    return {"tools": out, "errors": errors}


def _result_text(r):
    parts = []
    for c in r.get("content") or []:
        t = c.get("type")
        if t == "text":
            parts.append(c.get("text", ""))
        elif t in ("image", "audio"):
            parts.append("[%s returned: %s, %d bytes of base64, not shown]" % (t, c.get("mimeType", "?"), len(c.get("data") or "")))
        elif t == "resource":
            res = c.get("resource") or {}
            parts.append(res.get("text") or "[resource %s]" % res.get("uri", ""))
        elif t == "resource_link":
            parts.append("[resource link: %s %s]" % (c.get("name", ""), c.get("uri", "")))
    if not parts and r.get("structuredContent") is not None:
        parts.append(json.dumps(r["structuredContent"], ensure_ascii=False))
    text = "\n".join(p for p in parts if p) or "(no output)"
    if len(text) > RESULT_MAX:
        text = text[:RESULT_MAX] + "\n[cut: %d more characters]" % (len(text) - RESULT_MAX)
    return ("Error: " + text) if r.get("isError") else text


def call(name, args):
    for s in _load()["servers"]:
        if not s["enabled"]:
            continue
        c = _client(s)
        if c.tools is None:
            try:
                c.ensure()
            except McpError:
                continue
        for t in c.tools or []:
            if _fname(s, t["name"]) == name:
                started = time.time()
                try:
                    r = c.call(t["name"], args if isinstance(args, dict) else {})
                except McpError as e:
                    return {"ok": False, "server": s["name"], "tool": t["name"], "text": "Error: %s" % e}
                return {"ok": not r.get("isError"), "server": s["name"], "tool": t["name"],
                        "text": _result_text(r), "seconds": round(time.time() - started, 1)}
    return {"ok": False, "text": "Error: no tool named %s (its server may be off)" % name}


def handle(method, path, query, body):
    """(code, obj) for /mcp/*"""
    try:
        req = json.loads(body) if body else {}
        if not isinstance(req, dict):
            raise ValueError("a JSON object please")
        if method == "GET" and path == "/mcp/list":
            return 200, list_servers()
        if method == "GET" and path == "/mcp/get":
            return 200, get(query.get("id", ""))
        if method == "GET" and path == "/mcp/tools":
            return 200, tools()
        if method == "POST" and path == "/mcp/save":
            return 200, save(req.get("server"))
        if method == "POST" and path == "/mcp/delete":
            return 200, delete(req.get("id") or "")
        if method == "POST" and path == "/mcp/enable":
            return 200, enable(req.get("id") or "", bool(req.get("on")))
        if method == "POST" and path == "/mcp/connect":
            return 200, connect(req.get("id") or "")
        if method == "POST" and path == "/mcp/import":
            return 200, import_file(req.get("path"))
        if method == "POST" and path == "/mcp/call":
            return 200, call(str(req.get("name") or ""), req.get("arguments") or {})
        return 404, {"error": "no such route"}
    except ValueError as e:
        return 400, {"error": str(e)}
    except OSError as e:
        return 500, {"error": "%s: %s" % (type(e).__name__, e)}


if __name__ == "__main__":   # python ui/mcp.py: the servers and their tools, for a quick check
    json.dump(tools(), sys.stdout, indent=1)
    shutdown()
