#!/usr/bin/env python3
"""Small, dependency-free control plane for a local colibrì cluster.

The inference coordinator remains the C engine: it owns the token loop, KV
cache, and router.  This module only handles node registration and discovery;
expert activations use the binary TCP data plane implemented by ``glm``.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import socketserver
import struct
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


PROTOCOL_VERSION = 1
EXPERT_MAGIC = b"COLIEX01"
WEBSOCKET_PATH = "/v1/cluster/webgpu"


def parse_expert_batch(payload):
    """Decode the native expert batch protocol without touching activation bytes."""
    header = struct.Struct("!4I")
    if len(payload) < 8 + header.size or payload[:8] != EXPERT_MAGIC:
        raise ValueError("invalid expert batch magic")
    version, layer, hidden, count = header.unpack_from(payload, 8)
    if version != PROTOCOL_VERSION or count > 64 or not 1 <= hidden <= 65536:
        raise ValueError("invalid expert batch header")
    offset = 8 + header.size
    items = []
    for _ in range(count):
        if offset + 8 > len(payload):
            raise ValueError("truncated expert item")
        expert_id, rows = struct.unpack_from("!2I", payload, offset)
        offset += 8
        size = rows * hidden * 4
        if rows < 1 or rows > 65536 or offset + size > len(payload):
            raise ValueError("invalid expert activation payload")
        items.append({"expert_id": expert_id, "rows": rows,
                      "activations": payload[offset:offset + size]})
        offset += size
    if offset != len(payload):
        raise ValueError("trailing expert batch bytes")
    return {"version": version, "layer": layer, "hidden": hidden, "items": items}


def pack_expert_batch(batch):
    parts = [EXPERT_MAGIC, struct.pack("!4I", PROTOCOL_VERSION, batch["layer"],
                                       batch["hidden"], len(batch["items"]))]
    for item in batch["items"]:
        activations = item["activations"]
        expected = item["rows"] * batch["hidden"] * 4
        if len(activations) != expected:
            raise ValueError("activation byte count does not match item shape")
        parts.append(struct.pack("!2I", item["expert_id"], item["rows"]))
        parts.append(activations)
    return b"".join(parts)


def pack_expert_response(batch, outputs, status=0):
    parts = [EXPERT_MAGIC, struct.pack("!3I", PROTOCOL_VERSION, status, len(outputs))]
    for item, output in zip(batch["items"], outputs):
        if len(output) != item["rows"] * batch["hidden"] * 4:
            raise ValueError("expert output byte count does not match item shape")
        parts.append(struct.pack("!2I", item["expert_id"], item["rows"]))
        parts.append(output)
    return b"".join(parts)


def parse_expert_response(payload):
    if len(payload) < 8 + 12 or payload[:8] != EXPERT_MAGIC:
        raise ValueError("invalid expert response magic")
    version, status, count = struct.unpack_from("!3I", payload, 8)
    if version != PROTOCOL_VERSION or count > 64:
        raise ValueError("invalid expert response header")
    return {"version": version, "status": status, "count": count, "payload": payload[20:]}


def _ws_read_frame(sock):
    header = _recv_exact(sock, 2)
    first, second = header
    opcode = first & 0x0F
    masked = second & 0x80
    length = second & 0x7F
    if length == 126:
        length = struct.unpack("!H", _recv_exact(sock, 2))[0]
    elif length == 127:
        length = struct.unpack("!Q", _recv_exact(sock, 8))[0]
    if length > 64 << 20:
        raise ValueError("websocket frame too large")
    mask = _recv_exact(sock, 4) if masked else b""
    data = bytearray(_recv_exact(sock, length))
    if mask:
        for index in range(length):
            data[index] ^= mask[index % 4]
    return opcode, bytes(data)


def _recv_exact(sock, size):
    chunks = []
    while size:
        chunk = sock.recv(size)
        if not chunk:
            raise ConnectionError("socket closed")
        chunks.append(chunk)
        size -= len(chunk)
    return b"".join(chunks)


def _ws_frame(opcode, payload):
    length = len(payload)
    if length < 126:
        header = bytes((0x80 | opcode, length))
    elif length <= 0xFFFF:
        header = bytes((0x80 | opcode, 126)) + struct.pack("!H", length)
    else:
        header = bytes((0x80 | opcode, 127)) + struct.pack("!Q", length)
    return header + payload


class WebGPUConnection:
    def __init__(self, sock, node):
        self.sock = sock
        self.node = node
        self.lock = threading.Lock()
        self.closed = False

    def request(self, payload):
        with self.lock:
            if self.closed:
                raise ConnectionError("WebGPU worker is closed")
            self.sock.sendall(_ws_frame(2, payload))
            while True:
                opcode, data = _ws_read_frame(self.sock)
                if opcode == 2:
                    self.node["load"] = max(0, self.node.get("load", 1) - 1)
                    return data
                if opcode == 8:
                    raise ConnectionError("WebGPU worker closed")
                if opcode == 9:
                    self.sock.sendall(_ws_frame(10, data))

    def close(self):
        with self.lock:
            if not self.closed:
                self.closed = True
                try:
                    self.sock.sendall(_ws_frame(8, b""))
                except OSError:
                    pass
                try:
                    self.sock.close()
                except OSError:
                    pass


class ClusterRegistry:
    def __init__(self, stale_after=30.0):
        self.stale_after = float(stale_after)
        self._nodes = {}
        self._webgpu = {}
        self._lock = threading.Lock()

    def register(self, node):
        required = {"node_id", "host", "port", "role"}
        missing = sorted(required - set(node))
        if missing:
            raise ValueError("missing node fields: " + ", ".join(missing))
        role = str(node["role"])
        port = int(node["port"])
        if role != "webgpu" and not 1 <= port <= 65535:
            raise ValueError("port must be between 1 and 65535")
        if role not in ("expert", "dense", "webgpu", "coordinator"):
            raise ValueError("role must be expert, dense, webgpu, or coordinator")
        record = dict(node)
        record["protocol_version"] = PROTOCOL_VERSION
        record["port"] = port
        record["last_seen"] = time.time()
        with self._lock:
            self._nodes[str(node["node_id"])] = record
        return record

    def register_webgpu_connection(self, sock, hello):
        node = dict(hello)
        node.setdefault("role", "webgpu")
        node.setdefault("host", "browser")
        node.setdefault("port", 0)
        node.setdefault("device_type", "unknown")
        node.setdefault("precision", "f32")
        node.setdefault("expert_ids", ["*"])
        node["node_id"] = str(node.get("node_id") or f"webgpu-{id(sock)}")
        record = self.register(node)
        record["load"] = 0
        connection = WebGPUConnection(sock, record)
        with self._lock:
            self._webgpu[record["node_id"]] = connection
        return connection

    def unregister_webgpu(self, connection):
        with self._lock:
            node_id = connection.node["node_id"]
            if self._webgpu.get(node_id) is connection:
                del self._webgpu[node_id]
                self._nodes.pop(node_id, None)
        connection.close()

    def touch(self, node_id):
        with self._lock:
            node = self._nodes.get(str(node_id))
            if node:
                node["last_seen"] = time.time()

    def _webgpu_owner(self, layer, expert_id):
        key = f"{layer}:{expert_id}"
        with self._lock:
            candidates = []
            for connection in self._webgpu.values():
                node = connection.node
                experts = node.get("expert_ids", ["*"])
                if "*" in experts or key in experts or str(expert_id) in experts:
                    candidates.append(connection)
            if not candidates:
                return None
            return min(candidates, key=lambda item: item.node.get("load", 0))

    def dispatch_webgpu(self, payload):
        batch = parse_expert_batch(payload)
        groups = {}
        for item in batch["items"]:
            owner = self._webgpu_owner(batch["layer"], item["expert_id"])
            if owner is None:
                return pack_expert_response(batch, [], status=1)
            groups.setdefault(owner, []).append(item)
        results = {}
        for owner, items in groups.items():
            sub_batch = dict(batch, items=items)
            owner.node["load"] = owner.node.get("load", 0) + 1
            self.touch(owner.node["node_id"])
            response = parse_expert_response(owner.request(pack_expert_batch(sub_batch)))
            if response["status"] or response["count"] != len(items):
                return pack_expert_response(batch, [], status=1)
            results.update(_response_items(response["payload"], items, batch["hidden"]))
        outputs = [results[item["expert_id"]] for item in batch["items"]]
        return pack_expert_response(batch, outputs)

    def heartbeat(self, node_id):
        with self._lock:
            node = self._nodes.get(str(node_id))
            if node is None:
                raise KeyError(node_id)
            node["last_seen"] = time.time()
            return dict(node)

    def snapshot(self):
        now = time.time()
        with self._lock:
            nodes = [dict(node) for node in self._nodes.values()
                     if now - node["last_seen"] <= self.stale_after]
        nodes.sort(key=lambda node: (node["role"], node["node_id"]))
        return {"protocol_version": PROTOCOL_VERSION, "nodes": nodes,
                "webgpu_workers": len(self._webgpu)}

    def expert_endpoints(self):
        return [f"{node['host']}:{node['port']}"
                for node in self.snapshot()["nodes"] if node["role"] == "expert"]


def _response_items(payload, items, hidden):
    offset = 0
    outputs = {}
    for item in items:
        if offset + 8 > len(payload):
            raise ValueError("truncated WebGPU response")
        expert_id, rows = struct.unpack_from("!2I", payload, offset)
        offset += 8
        size = rows * hidden * 4
        if expert_id != item["expert_id"] or rows != item["rows"] or offset + size > len(payload):
            raise ValueError("invalid WebGPU response item")
        outputs[expert_id] = payload[offset:offset + size]
        offset += size
    if offset != len(payload):
        raise ValueError("trailing WebGPU response bytes")
    return outputs


class _Handler(BaseHTTPRequestHandler):
    server_version = "colibri-cluster/1"

    def log_message(self, *_args):
        return

    def _json(self, status, value):
        payload = json.dumps(value, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _body(self):
        length = int(self.headers.get("Content-Length", "0"))
        if length > 1 << 20:
            raise ValueError("request body is too large")
        return json.loads(self.rfile.read(length) or b"{}")

    def do_GET(self):  # noqa: N802 - stdlib handler API
        if self.path == WEBSOCKET_PATH and self.headers.get("Upgrade", "").lower() == "websocket":
            self._webgpu_websocket()
            return
        if self.path in ("/health", "/v1/cluster/topology"):
            self._json(200, self.server.registry.snapshot())
            return
        self._json(404, {"error": "not found"})

    def _webgpu_websocket(self):
        key = self.headers.get("Sec-WebSocket-Key")
        if not key:
            self._json(400, {"error": "missing Sec-WebSocket-Key"})
            return
        accept = base64.b64encode(hashlib.sha1(
            (key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()).digest()).decode()
        self.send_response(101, "Switching Protocols")
        self.send_header("Upgrade", "websocket")
        self.send_header("Connection", "Upgrade")
        self.send_header("Sec-WebSocket-Accept", accept)
        self.end_headers()
        self.wfile.flush()
        connection = None
        try:
            opcode, payload = _ws_read_frame(self.connection)
            if opcode != 1:
                raise ValueError("WebGPU worker must begin with a JSON hello")
            hello = json.loads(payload.decode("utf-8"))
            if hello.get("role", "webgpu") != "webgpu":
                raise ValueError("worker role must be webgpu")
            connection = self.server.registry.register_webgpu_connection(self.connection, hello)
            self.server.registry.touch(connection.node["node_id"])
            while not connection.closed:
                self.server.registry.touch(connection.node["node_id"])
                time.sleep(1)
        except (ConnectionError, OSError, ValueError, json.JSONDecodeError):
            pass
        finally:
            if connection is not None:
                self.server.registry.unregister_webgpu(connection)

    def do_POST(self):  # noqa: N802 - stdlib handler API
        try:
            body = self._body()
            if self.path == "/v1/cluster/register":
                self._json(200, self.server.registry.register(body))
            elif self.path == "/v1/cluster/heartbeat":
                self._json(200, self.server.registry.heartbeat(body["node_id"]))
            else:
                self._json(404, {"error": "not found"})
        except (KeyError, ValueError, TypeError, json.JSONDecodeError) as error:
            self._json(400, {"error": str(error)})


class _WebGPUProxyHandler(socketserver.BaseRequestHandler):
    def handle(self):
        while True:
            try:
                header = _recv_exact(self.request, 24)
                if header[:8] != EXPERT_MAGIC:
                    return
                version, layer, hidden, count = struct.unpack("!4I", header[8:])
                if version != PROTOCOL_VERSION or count > 64 or not 1 <= hidden <= 65536:
                    return
                parts = [header]
                for _ in range(count):
                    item_header = _recv_exact(self.request, 8)
                    expert_id, rows = struct.unpack("!2I", item_header)
                    if rows < 1 or rows > 65536:
                        return
                    parts.extend((item_header, _recv_exact(self.request, rows * hidden * 4)))
                response = self.server.registry.dispatch_webgpu(b"".join(parts))
                self.request.sendall(response)
            except (ConnectionError, OSError, ValueError, KeyError):
                return


class _ThreadingTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


class ClusterServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, registry):
        super().__init__(address, _Handler)
        self.registry = registry


def serve(host="127.0.0.1", port=8765, stale_after=30.0, data_port=8766):
    registry = ClusterRegistry(stale_after)
    server = ClusterServer((host, port), registry)
    proxy = _ThreadingTCPServer((host, data_port), _WebGPUProxyHandler)
    proxy.registry = registry
    threading.Thread(target=proxy.serve_forever, name="colibri-webgpu-proxy", daemon=True).start()
    print(f"colibri cluster coordinator listening on http://{host}:{port} · WebGPU data port {data_port}", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        proxy.shutdown()
        proxy.server_close()


def register(coordinator, node):
    url = coordinator.rstrip("/") + "/v1/cluster/register"
    request = Request(url, data=json.dumps(node).encode(),
                     headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(request, timeout=5) as response:
        return json.load(response)


def heartbeat(coordinator, node_id):
    url = coordinator.rstrip("/") + "/v1/cluster/heartbeat"
    request = Request(url, data=json.dumps({"node_id": node_id}).encode(),
                     headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(request, timeout=5) as response:
        return json.load(response)


def discover_workers(coordinator):
    url = coordinator.rstrip("/") + "/v1/cluster/topology"
    with urlopen(url, timeout=5) as response:
        topology = json.load(response)
    return [f"{node['host']}:{int(node['port'])}"
            for node in topology.get("nodes", []) if node.get("role") == "expert"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--stale-after", type=float, default=30.0)
    parser.add_argument("--data-port", type=int, default=8766)
    args = parser.parse_args()
    serve(args.host, args.port, args.stale_after, args.data_port)


if __name__ == "__main__":
    main()
