#!/usr/bin/env python3
"""Bridge browser WebGPU workers to colibri's binary expert data plane."""

import argparse
import base64
import hashlib
import json
import socketserver
import struct
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


MAGIC = b"COLIEX01"
VERSION = 1
WEBSOCKET_PATH = "/v1/webgpu"


def _recv_exact(sock, size):
    chunks = []
    while size:
        chunk = sock.recv(size)
        if not chunk:
            raise ConnectionError("socket closed")
        chunks.append(chunk)
        size -= len(chunk)
    return b"".join(chunks)


def _ws_read_frame(sock):
    first, second = _recv_exact(sock, 2)
    opcode = first & 0x0F
    masked = second & 0x80
    length = second & 0x7F
    if length == 126:
        length = struct.unpack("!H", _recv_exact(sock, 2))[0]
    elif length == 127:
        length = struct.unpack("!Q", _recv_exact(sock, 8))[0]
    if length > 64 << 20:
        raise ValueError("WebSocket frame too large")
    mask = _recv_exact(sock, 4) if masked else b""
    data = bytearray(_recv_exact(sock, length))
    for index, value in enumerate(mask):
        for offset in range(index, length, 4):
            data[offset] ^= value
    return opcode, bytes(data)


def _ws_frame(opcode, payload):
    length = len(payload)
    if length < 126:
        return bytes((0x80 | opcode, length)) + payload
    if length <= 0xFFFF:
        return bytes((0x80 | opcode, 126)) + struct.pack("!H", length) + payload
    return bytes((0x80 | opcode, 127)) + struct.pack("!Q", length) + payload


def parse_batch(payload):
    if len(payload) < 28 or payload[:8] != MAGIC:
        raise ValueError("invalid expert batch magic")
    version, layer, hidden, intermediate, count = struct.unpack_from("!5I", payload, 8)
    if (version != VERSION or count > 64 or not 1 <= hidden <= 65536
            or not 1 <= intermediate <= 1 << 20):
        raise ValueError("invalid expert batch header")
    offset, items = 28, []
    for _ in range(count):
        if offset + 8 > len(payload):
            raise ValueError("truncated expert item")
        expert_id, rows = struct.unpack_from("!2I", payload, offset)
        offset += 8
        if not 1 <= rows <= 65536:
            raise ValueError("invalid expert rows")
        size = rows * hidden * 4
        if offset + size > len(payload):
            raise ValueError("truncated activation payload")
        items.append({"expert_id": expert_id, "rows": rows,
                      "activations": payload[offset:offset + size]})
        offset += size
    if offset != len(payload):
        raise ValueError("trailing expert bytes")
    return {"version": version, "layer": layer, "hidden": hidden,
            "intermediate": intermediate, "items": items}


def pack_batch(batch):
    parts = [MAGIC, struct.pack("!5I", VERSION, batch["layer"], batch["hidden"],
                                batch["intermediate"], len(batch["items"]))]
    for item in batch["items"]:
        expected = item["rows"] * batch["hidden"] * 4
        if len(item["activations"]) != expected:
            raise ValueError("activation byte count does not match shape")
        parts += [struct.pack("!2I", item["expert_id"], item["rows"]), item["activations"]]
    return b"".join(parts)


def pack_response(batch, outputs, status=0):
    parts = [MAGIC, struct.pack("!3I", VERSION, status, len(outputs))]
    for item, output in zip(batch["items"], outputs):
        expected = item["rows"] * batch["hidden"] * 4
        if len(output) != expected:
            raise ValueError("output byte count does not match shape")
        parts += [struct.pack("!2I", item["expert_id"], item["rows"]), output]
    return b"".join(parts)


def parse_response(payload):
    if len(payload) < 20 or payload[:8] != MAGIC:
        raise ValueError("invalid expert response")
    version, status, count = struct.unpack_from("!3I", payload, 8)
    if version != VERSION or count > 64:
        raise ValueError("invalid expert response header")
    return version, status, count, payload[20:]


class WebGPUConnection:
    def __init__(self, sock, node):
        self.sock, self.node = sock, node
        self.lock = threading.Lock()

    def request(self, payload):
        with self.lock:
            self.node["load"] = self.node.get("load", 0) + 1
            self.sock.sendall(_ws_frame(2, payload))
            try:
                while True:
                    opcode, data = _ws_read_frame(self.sock)
                    if opcode == 2:
                        return data
                    if opcode == 9:
                        self.sock.sendall(_ws_frame(10, data))
                    elif opcode == 8:
                        raise ConnectionError("WebGPU worker closed")
            finally:
                self.node["load"] = max(0, self.node.get("load", 1) - 1)

    def close(self):
        try:
            self.sock.sendall(_ws_frame(8, b""))
        except OSError:
            pass
        self.sock.close()


class WebGPURegistry:
    def __init__(self):
        self._workers = {}
        self._lock = threading.Lock()

    def register(self, sock, hello):
        if hello.get("role", "webgpu") != "webgpu":
            raise ValueError("worker role must be webgpu")
        node_id = str(hello.get("node_id") or f"webgpu-{id(sock)}")
        node = dict(hello, node_id=node_id, role="webgpu", load=0)
        connection = WebGPUConnection(sock, node)
        with self._lock:
            self._workers[node_id] = connection
        return connection

    def unregister(self, connection):
        with self._lock:
            if self._workers.get(connection.node["node_id"]) is connection:
                del self._workers[connection.node["node_id"]]
        connection.close()

    def _owner(self, layer, expert_id):
        key = f"{layer}:{expert_id}"
        with self._lock:
            candidates = [worker for worker in self._workers.values()
                          if "*" in worker.node.get("expert_ids", ["*"])
                          or key in worker.node.get("expert_ids", [])
                          or str(expert_id) in worker.node.get("expert_ids", [])]
        return min(candidates, key=lambda worker: worker.node.get("load", 0)) if candidates else None

    def dispatch(self, payload):
        batch = parse_batch(payload)
        grouped = {}
        for item in batch["items"]:
            owner = self._owner(batch["layer"], item["expert_id"])
            if owner is None:
                return pack_response(batch, [], status=1)
            grouped.setdefault(owner, []).append(item)
        results = {}
        for owner, items in grouped.items():
            sub_batch = dict(batch, items=items)
            version, status, count, rest = parse_response(owner.request(pack_batch(sub_batch)))
            if status or count != len(items):
                return pack_response(batch, [], status=1)
            offset = 0
            for item in items:
                if offset + 8 > len(rest):
                    raise ValueError("truncated WebGPU output")
                expert_id, rows = struct.unpack_from("!2I", rest, offset)
                offset += 8
                size = rows * batch["hidden"] * 4
                if expert_id != item["expert_id"] or rows != item["rows"] or offset + size > len(rest):
                    raise ValueError("invalid WebGPU output shape")
                results[expert_id] = rest[offset:offset + size]
                offset += size
            if offset != len(rest):
                raise ValueError("trailing WebGPU output")
        return pack_response(batch, [results[item["expert_id"]] for item in batch["items"]])


class _ControlHandler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        return

    def do_GET(self):  # noqa: N802 - stdlib handler API
        if self.path != WEBSOCKET_PATH or self.headers.get("Upgrade", "").lower() != "websocket":
            self.send_error(404)
            return
        key = self.headers.get("Sec-WebSocket-Key")
        if not key:
            self.send_error(400, "missing Sec-WebSocket-Key")
            return
        accept = base64.b64encode(hashlib.sha1(
            (key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()).digest()).decode()
        self.send_response(101, "Switching Protocols")
        self.send_header("Upgrade", "websocket")
        self.send_header("Connection", "Upgrade")
        self.send_header("Sec-WebSocket-Accept", accept)
        self.end_headers()
        connection = None
        try:
            opcode, payload = _ws_read_frame(self.connection)
            if opcode != 1:
                raise ValueError("WebGPU worker must begin with JSON hello")
            connection = self.server.registry.register(self.connection, json.loads(payload))
            while True:
                # The data proxy owns reads after registration; reading here
                # would race the binary response path on the same socket.
                time.sleep(1)
        except (ConnectionError, OSError, ValueError, json.JSONDecodeError):
            pass
        finally:
            if connection:
                self.server.registry.unregister(connection)


class _ControlServer(ThreadingHTTPServer):
    daemon_threads = True


class _ProxyHandler(socketserver.BaseRequestHandler):
    def handle(self):
        try:
            while True:
                header = _recv_exact(self.request, 28)
                if header[:8] != MAGIC:
                    raise ValueError("invalid expert batch magic")
                version, layer, hidden, intermediate, count = struct.unpack("!5I", header[8:])
                if (version != VERSION or count > 64 or not 1 <= hidden <= 65536
                        or not 1 <= intermediate <= 1 << 20):
                    raise ValueError("invalid expert batch header")
                batch_parts = [header]
                for _ in range(count):
                    item_header = _recv_exact(self.request, 8)
                    _, rows = struct.unpack("!2I", item_header)
                    if not 1 <= rows <= 65536:
                        raise ValueError("invalid expert rows")
                    batch_parts += [item_header, _recv_exact(self.request, rows * hidden * 4)]
                self.request.sendall(self.server.registry.dispatch(b"".join(batch_parts)))
        except (ConnectionError, OSError, ValueError):
            return


class _ProxyServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


def serve(host="127.0.0.1", control_port=8765, data_port=8766):
    registry = WebGPURegistry()
    control = _ControlServer((host, control_port), _ControlHandler)
    control.registry = registry
    proxy = _ProxyServer((host, data_port), _ProxyHandler)
    proxy.registry = registry
    threading.Thread(target=proxy.serve_forever, daemon=True).start()
    print(f"WebGPU coordinator: ws://{host}:{control_port}{WEBSOCKET_PATH} · data {data_port}", flush=True)
    try:
        control.serve_forever()
    finally:
        control.server_close()
        proxy.shutdown()
        proxy.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--control-port", type=int, default=8765)
    parser.add_argument("--data-port", type=int, default=8766)
    args = parser.parse_args()
    serve(args.host, args.control_port, args.data_port)
