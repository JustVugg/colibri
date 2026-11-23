"""MCP server (stdio, standard library only) that bridges Colibri to the Blender MCP add-on.

The Blender add-on does NOT speak MCP. It runs a plain TCP socket server inside Blender
(localhost:9876) that takes one JSON request per connection, terminated by a NUL byte:
    {"type": "execute", "code": "<python>", "strict_json": false}\0
and answers with one JSON object terminated by a NUL byte. The Python code runs inside Blender
and must put its output in a dict called `result`. This file translates MCP (stdio, JSON-RPC,
newline-delimited) into those socket calls.

Prerequisite: in Blender, open the sidebar (N) > MCP tab and click Start (it listens on localhost:9876).
Register it in colibri's MCP config as a stdio server: a Python interpreter plus this file's absolute path.
See docs/MCP_BLENDER.md.
"""
import json
import socket
import sys

HOST = "127.0.0.1"
PORT = 9876
TIMEOUT = 120
PROTOCOL_FALLBACK = "2025-06-18"

TOOLS = [
    {
        "name": "blender_status",
        "description": "Check that Blender's MCP add-on is listening and report the Blender version and open file.",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "blender_execute",
        "description": (
            "Run Python code inside Blender (bpy is available). Put the output in a dict named `result`, "
            "for example: import bpy; result = {'objects': [o.name for o in bpy.data.objects]}. "
            "Print statements are returned as stdout."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"code": {"type": "string", "description": "Python code to run inside Blender."}},
            "required": ["code"],
            "additionalProperties": False,
        },
    },
]


def send_to_blender(payload):
    """Send one request to the Blender add-on and return its decoded JSON response."""
    data = (json.dumps(payload) + "\0").encode("utf-8")
    with socket.create_connection((HOST, PORT), timeout=TIMEOUT) as sock:
        sock.sendall(data)
        buf = bytearray()
        while True:
            chunk = sock.recv(65536)
            if not chunk:
                break
            buf.extend(chunk)
            if b"\0" in buf:
                break
    raw = bytes(buf[: buf.index(b"\0")]) if b"\0" in buf else bytes(buf)
    if not raw:
        raise RuntimeError("Blender closed the connection without a response")
    return json.loads(raw.decode("utf-8"))


def run_blender_code(code):
    return send_to_blender({"type": "execute", "code": code, "strict_json": False})


def tool_text(value, is_error=False):
    return {"content": [{"type": "text", "text": value}], "isError": is_error}


def call_tool(name, args):
    try:
        if name == "blender_status":
            code = (
                "import bpy\n"
                "result = {'ok': True, 'blender': bpy.app.version_string, "
                "'file': bpy.data.filepath or '(unsaved)', 'objects': len(bpy.data.objects)}"
            )
        elif name == "blender_execute":
            code = str(args.get("code", ""))
            if not code.strip():
                return tool_text("Error: empty code", True)
        else:
            return tool_text("Error: unknown tool " + name, True)

        resp = run_blender_code(code)
    except (ConnectionRefusedError, socket.timeout, OSError) as ex:
        return tool_text(
            "Blender is not listening on %s:%d (%s). Open Blender, press N for the sidebar, "
            "go to the MCP tab and click Start. Then try again." % (HOST, PORT, ex),
            True,
        )
    except Exception as ex:  # keep the MCP session alive whatever happens
        return tool_text("Error talking to Blender: %s" % ex, True)

    if resp.get("status") != "ok":
        msg = resp.get("message") or json.dumps(resp)
        return tool_text("Blender error:\n" + msg, True)

    out = {"result": resp.get("result")}
    if resp.get("stdout"):
        out["stdout"] = resp["stdout"]
    if resp.get("stderr"):
        out["stderr"] = resp["stderr"]
    return tool_text(json.dumps(out, indent=2))


def reply(msg_id, result=None, error=None):
    msg = {"jsonrpc": "2.0", "id": msg_id}
    if error is not None:
        msg["error"] = error
    else:
        msg["result"] = result if result is not None else {}
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def handle(msg):
    method = msg.get("method")
    msg_id = msg.get("id")
    params = msg.get("params") or {}

    if msg_id is None:  # notifications (e.g. notifications/initialized) need no reply
        return
    if method == "initialize":
        reply(msg_id, {
            "protocolVersion": params.get("protocolVersion", PROTOCOL_FALLBACK),
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "blender-bridge", "version": "1.0.0"},
        })
    elif method == "ping":
        reply(msg_id, {})
    elif method == "tools/list":
        reply(msg_id, {"tools": TOOLS})
    elif method == "tools/call":
        reply(msg_id, call_tool(params.get("name", ""), params.get("arguments") or {}))
    else:
        reply(msg_id, error={"code": -32601, "message": "Method not found: %s" % method})


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        try:
            handle(msg)
        except Exception as ex:
            if msg.get("id") is not None:
                reply(msg["id"], error={"code": -32603, "message": str(ex)})


if __name__ == "__main__":
    main()
