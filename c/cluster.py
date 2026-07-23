#!/usr/bin/env python3
"""Registration and discovery control plane for local expert workers."""

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import Request, urlopen


PROTOCOL_VERSION = 1


class ClusterRegistry:
    def __init__(self, stale_after=30.0):
        self.stale_after = float(stale_after)
        self._nodes = {}
        self._lock = threading.Lock()

    def register(self, node):
        required = {"node_id", "host", "port", "role"}
        missing = sorted(required - set(node))
        if missing:
            raise ValueError("missing node fields: " + ", ".join(missing))
        port = int(node["port"])
        if not 1 <= port <= 65535:
            raise ValueError("port must be between 1 and 65535")
        role = str(node["role"])
        if role not in ("expert", "dense", "coordinator"):
            raise ValueError("role must be expert, dense, or coordinator")
        record = dict(node)
        record.update(protocol_version=PROTOCOL_VERSION, port=port, last_seen=time.time())
        with self._lock:
            self._nodes[str(node["node_id"])] = record
        return record

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
        return {"protocol_version": PROTOCOL_VERSION, "nodes": nodes}

    def expert_endpoints(self):
        return [f"{node['host']}:{node['port']}"
                for node in self.snapshot()["nodes"] if node["role"] == "expert"]


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
        if self.path in ("/health", "/v1/cluster/topology"):
            self._json(200, self.server.registry.snapshot())
            return
        self._json(404, {"error": "not found"})

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


class ClusterServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, registry):
        super().__init__(address, _Handler)
        self.registry = registry


def serve(host="127.0.0.1", port=8765, stale_after=30.0):
    server = ClusterServer((host, port), ClusterRegistry(stale_after))
    print(f"colibri cluster coordinator listening on http://{host}:{port}", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


def register(coordinator, node):
    request = Request(coordinator.rstrip("/") + "/v1/cluster/register",
                      data=json.dumps(node).encode(),
                      headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(request, timeout=5) as response:
        return json.load(response)


def heartbeat(coordinator, node_id):
    request = Request(coordinator.rstrip("/") + "/v1/cluster/heartbeat",
                      data=json.dumps({"node_id": node_id}).encode(),
                      headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(request, timeout=5) as response:
        return json.load(response)


def discover_workers(coordinator):
    with urlopen(coordinator.rstrip("/") + "/v1/cluster/topology", timeout=5) as response:
        topology = json.load(response)
    return [f"{node['host']}:{int(node['port'])}"
            for node in topology.get("nodes", []) if node.get("role") == "expert"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--stale-after", type=float, default=30.0)
    args = parser.parse_args()
    serve(args.host, args.port, args.stale_after)


if __name__ == "__main__":
    main()
