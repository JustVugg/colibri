# Blender in colibri's chat (MCP)

colibri can drive a running Blender through the **Blender Lab MCP add-on**
(Blender 5.1 or newer). Two tools are exposed:

- `blender_status`: checks Blender is listening and reports the Blender version, the open file and the object count.
- `blender_execute`: runs Python inside Blender (`bpy` is available). Put the output in a dict named `result`; printed text comes back as `stdout`.

## Why a bridge is needed

The add-on does not speak MCP. It runs a plain TCP socket server inside Blender
on `localhost:9876`, taking one NUL-terminated JSON request per connection.
`mcp-servers/blender/blender_bridge_mcp.py` is a small stdio MCP server
(standard library only) that translates between the two. Nothing else needs
installing beyond a Python 3.10+ interpreter.

## Setup

1. Install the Blender MCP add-on and enable it.
2. Open Blender, press **N** for the sidebar, go to the **MCP** tab and click **Start**. Blender must stay open; the server stops when Blender closes.
3. Add a stdio server to colibri's MCP config (`~/colibri-mcp.json` by default), with absolute paths:

```json
{
  "servers": [
    {
      "id": "blender",
      "name": "Blender",
      "description": "Blender via the Blender Lab MCP add-on (Blender must have MCP > Start clicked).",
      "enabled": true,
      "connection": {
        "type": "stdio",
        "command": "C:\\path\\to\\python.exe",
        "args": ["-I", "C:\\path\\to\\colibri\\mcp-servers\\blender\\blender_bridge_mcp.py"],
        "env": {},
        "timeoutMs": 120000
      }
    }
  ]
}
```

On Linux or macOS use `python3` and `/path/to/colibri/mcp-servers/blender/blender_bridge_mcp.py`.

4. In the chat, ask for it, for example: *"Use the Blender status tool and tell me what it returns."*

## Troubleshooting

- **"Blender is not listening on 127.0.0.1:9876"**: Blender is closed, or MCP > Start has not been clicked. Nothing else is wrong with the config.
- **Connection works but `blender_execute` errors**: the error text is Blender's traceback. Check the code; the `result` variable must be a dict.
- Scene changes are not saved automatically. Save the `.blend` file yourself.

## Protocol details

- Transport: stdio, newline-delimited JSON-RPC 2.0.
- Methods: `initialize`, `ping`, `tools/list`, `tools/call`. Notifications are accepted and ignored.
- Add-on request: `{"type": "execute", "code": "...", "strict_json": false}` followed by a NUL byte; the response is one JSON object followed by a NUL byte (`status`, `result`, `stdout`, `stderr`, or `message` on error).
