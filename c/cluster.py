#!/usr/bin/env python3
"""Registration and discovery control plane for local expert workers."""

import argparse
import hmac
import json
import os
import sys
import threading
import time
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import openai_server


PROTOCOL_VERSION = 1
# A shared secret for the coordinator. Set on the coordinator and on every worker (the workers'
# register and heartbeat calls read it from here), and registration then needs it. Read from the
# environment, not a flag: a command-line secret shows in the process list.
TOKEN_ENV = "COLI_CLUSTER_TOKEN"
LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")


class ClusterRegistry:
    def __init__(self, stale_after=30.0):
        self.stale_after = float(stale_after)
        self._nodes = {}
        self._last_seen = {}
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
        # The registry keys by str(node_id) and heartbeat looks it up the same
        # way, so a numeric node_id has always registered. Store the string in
        # the record too, or snapshot() sorts it against a string node_id and
        # every topology/health request dies with a TypeError.
        node_id = str(node["node_id"])
        record = dict(node)
        record.update(node_id=node_id, protocol_version=PROTOCOL_VERSION, port=port,
                      last_seen=time.time())
        with self._lock:
            self._nodes[node_id] = record
            self._last_seen[node_id] = time.monotonic()
        return record

    def heartbeat(self, node_id):
        with self._lock:
            node = self._nodes.get(str(node_id))
            if node is None:
                raise KeyError(node_id)
            node["last_seen"] = time.time()
            self._last_seen[str(node_id)] = time.monotonic()
            return dict(node)

    def snapshot(self):
        now = time.monotonic()
        with self._lock:
            nodes = [dict(node) for node_id, node in self._nodes.items()
                     if now - self._last_seen[node_id] <= self.stale_after]
        nodes.sort(key=lambda node: (node["role"], node["node_id"]))
        return {"protocol_version": PROTOCOL_VERSION, "nodes": nodes}

    def expert_endpoints(self):
        return [f"{node['host']}:{node['port']}"
                for node in self.snapshot()["nodes"] if node["role"] == "expert"]


class _Handler(openai_server.APIHandler):
    # The original registry served HTTP/1.0: one request per connection, no
    # keep-alive. Keep that wire shape; the shared APIHandler machinery (Host
    # guard, body cap, deadline reader, connection bookkeeping) still applies.
    protocol_version = "HTTP/1.0"
    server_version = "colibri-cluster/1"

    def log_message(self, *_args):
        return

    def _fail(self, error):
        self.send_json(error.status, {"error": error.message})

    def do_GET(self):  # noqa: N802 - stdlib handler API
        try:
            self._check_host()
        except openai_server.APIError as error:
            self._fail(error)
            return
        if self.path in ("/health", "/v1/cluster/topology"):
            self.send_json(200, self.server.registry.snapshot())
            return
        self.send_json(404, {"error": "not found"})

    def _check_token(self):
        """registration and heartbeat need the shared secret when the coordinator has one. The
        Host guard is not authentication: it stops a web page from re-pointing a name here, but
        any host that can reach the port can send `Host: 127.0.0.1`."""
        token = getattr(self.server, "token", None)
        if not token:
            return
        sent = self.headers.get("Authorization", "")
        if not hmac.compare_digest(sent.encode("utf-8"), f"Bearer {token}".encode("utf-8")):
            raise openai_server.APIError(
                401, f"registration needs the cluster token: send 'Authorization: Bearer <${TOKEN_ENV}>'")

    def do_POST(self):  # noqa: N802 - stdlib handler API
        try:
            self._check_host()
            self._check_token()
            body = self.read_json()
            if self.path == "/v1/cluster/register":
                self.send_json(200, self.server.registry.register(body))
            elif self.path == "/v1/cluster/heartbeat":
                self.send_json(200, self.server.registry.heartbeat(body["node_id"]))
            else:
                self.send_json(404, {"error": "not found"})
        except openai_server.APIError as error:
            self._fail(error)
        except (KeyError, ValueError, TypeError) as error:
            self.send_json(400, {"error": str(error)})


class ClusterServer(openai_server.APIServer):
    def __init__(self, address, registry, allowed_hosts=(), token=None):
        ThreadingHTTPServer.__init__(self, address, _Handler)
        self.registry = registry
        self.token = token or None   # None or "": registration is open, as before
        # Shared APIHandler plumbing reads these off the server. The registry has
        # no API key or CORS surface, and its Host guard stays loopback + bind
        # address (the same default openai_server enforces). A cross-host worker
        # registers from a LAN IP, so the operator opts those hosts in via
        # --allowed-host (the same #597 escape hatch coli serve exposes); the
        # default stays loopback + bind address.
        self.cors_origins = ()
        self.allowed_hosts = tuple(
            h.strip().lower() for h in allowed_hosts if h and h.strip())
        # Connection-cap bookkeeping for the inherited APIServer.process_request /
        # _release / close_request. There is no engine or model to carry, so
        # initialise the tracking state directly rather than APIServer.__init__.
        self._conn_lock = threading.Lock()
        self._conn_live = 0
        self._conn_by_ip = {}
        self._conn_owner = {}


def _endpoint(coordinator, path):
    return coordinator.rstrip("/") + "/v1/cluster/" + path


def _auth_headers():
    token = os.environ.get(TOKEN_ENV)
    return {"Authorization": f"Bearer {token}"} if token else {}


def _post(request):
    try:
        with urlopen(request, timeout=5) as response:
            return json.load(response)
    except HTTPError as error:
        if error.code == 401:
            raise OSError(f"the coordinator refused this node: set {TOKEN_ENV} to the coordinator's "
                          f"cluster token") from error
        raise


def exposure_warning(host, token):
    """what to tell the operator when the coordinator listens beyond this machine with an open
    registration, or None"""
    if token or str(host).strip("[]").lower() in LOOPBACK_HOSTS:
        return None
    return (f"WARNING: the coordinator listens on {host} and {TOKEN_ENV} is not set, so any host that "
            f"can reach it can register as an expert worker; set {TOKEN_ENV} here and on every worker")


def serve(host="127.0.0.1", port=8765, stale_after=30.0, allowed_hosts=(), token=None):
    if allowed_hosts and "*" in allowed_hosts:
        print("WARNING: --allowed-host '*' accepts ANY Host header "
              "(DNS-rebinding guard disabled)", file=sys.stderr)
    token = token if token is not None else os.environ.get(TOKEN_ENV)
    warning = exposure_warning(host, token)
    if warning:
        print(warning, file=sys.stderr)
    server = ClusterServer((host, port), ClusterRegistry(stale_after), allowed_hosts, token)
    print(f"colibri cluster coordinator listening on http://{host}:{port}", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


def register(coordinator, node):
    request = Request(_endpoint(coordinator, "register"),
                      data=json.dumps(node).encode(),
                      headers={"Content-Type": "application/json", **_auth_headers()}, method="POST")
    return _post(request)


def heartbeat(coordinator, node_id):
    request = Request(_endpoint(coordinator, "heartbeat"),
                      data=json.dumps({"node_id": node_id}).encode(),
                      headers={"Content-Type": "application/json", **_auth_headers()}, method="POST")
    return _post(request)


def discover_workers(coordinator):
    with urlopen(_endpoint(coordinator, "topology"), timeout=5) as response:
        topology = json.load(response)
    return [f"{node['host']}:{int(node['port'])}"
            for node in topology.get("nodes", []) if node.get("role") == "expert"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--stale-after", type=float, default=30.0)
    parser.add_argument("--allowed-host", action="append",
                        default=[h.strip() for h in os.environ.get("COLI_ALLOWED_HOSTS", "").split(",") if h.strip()],
                        help="additional Host header accepted by the DNS-rebinding guard; repeat as needed")
    args = parser.parse_args()
    serve(args.host, args.port, args.stale_after, args.allowed_host)


if __name__ == "__main__":
    main()
