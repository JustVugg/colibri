import importlib.util
import json
import sys
import threading
import types
import unittest
from http.client import HTTPConnection
from importlib.machinery import SourceFileLoader
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cluster import ClusterRegistry, ClusterServer, PROTOCOL_VERSION


class ClusterRegistryTests(unittest.TestCase):
    def test_registers_and_discovers_expert_nodes(self):
        registry = ClusterRegistry()
        registry.register({"node_id": "mac-a", "host": "10.0.0.2", "port": 9100,
                           "role": "expert", "layers": "all"})
        registry.register({"node_id": "mac-b", "host": "10.0.0.3", "port": 9101,
                           "role": "dense", "layers": "38-75"})
        self.assertEqual(registry.expert_endpoints(), ["10.0.0.2:9100"])
        self.assertEqual(registry.snapshot()["protocol_version"], PROTOCOL_VERSION)

    def test_numeric_node_id_does_not_break_discovery(self):
        # A worker that sends "node_id": 1 registers fine (the registry keys
        # by str(node_id) and heartbeat looks it up the same way), so the
        # snapshot must not die sorting an int against a string node_id.
        registry = ClusterRegistry()
        record = registry.register({"node_id": 1, "host": "10.0.0.2", "port": 9100,
                                    "role": "expert"})
        registry.register({"node_id": "mac-b", "host": "10.0.0.3", "port": 9101,
                           "role": "expert"})
        self.assertEqual(registry.expert_endpoints(), ["10.0.0.2:9100", "10.0.0.3:9101"])
        self.assertEqual(registry.heartbeat(record["node_id"])["host"], "10.0.0.2")
        self.assertEqual(registry.heartbeat(1)["host"], "10.0.0.2")

    def test_http_topology_survives_a_numeric_node_id(self):
        server = ClusterServer(("127.0.0.1", 0), ClusterRegistry())
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            for node_id in (1, "mac-b"):
                conn = HTTPConnection(*server.server_address)
                body = json.dumps({"node_id": node_id, "host": "127.0.0.1",
                                   "port": 9100, "role": "expert"})
                conn.request("POST", "/v1/cluster/register", body,
                             {"Content-Type": "application/json"})
                self.assertEqual(conn.getresponse().status, 200)
                conn.close()
            for path in ("/v1/cluster/topology", "/health"):
                conn = HTTPConnection(*server.server_address)
                conn.request("GET", path)
                response = conn.getresponse()
                self.assertEqual(response.status, 200, path)
                self.assertEqual(len(json.loads(response.read())["nodes"]), 2, path)
                conn.close()
        finally:
            server.shutdown()
            server.server_close()

    def test_http_topology_registration_and_heartbeat(self):
        server = ClusterServer(("127.0.0.1", 0), ClusterRegistry())
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            conn = HTTPConnection(*server.server_address)
            body = json.dumps({"node_id": "mac-a", "host": "127.0.0.1",
                               "port": 9100, "role": "expert"})
            conn.request("POST", "/v1/cluster/register", body,
                         {"Content-Type": "application/json"})
            self.assertEqual(conn.getresponse().status, 200)
            conn.request("POST", "/v1/cluster/heartbeat", json.dumps({"node_id": "mac-a"}),
                         {"Content-Type": "application/json"})
            self.assertEqual(conn.getresponse().status, 200)
            conn.request("GET", "/v1/cluster/topology")
            response = conn.getresponse()
            self.assertEqual(response.status, 200)
            self.assertEqual(len(json.loads(response.read())["nodes"]), 1)
            conn.close()
        finally:
            server.shutdown()
            server.server_close()


def _start(port=0):
    server = ClusterServer(("127.0.0.1", port), ClusterRegistry())
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


class ClusterWorkerReRegisterTests(unittest.TestCase):
    """The registry lives in the coordinator's memory. A worker that keeps
    running across a coordinator restart must register again, or every
    heartbeat is refused and `coli serve --cluster-coordinator` finds no
    live expert workers until the worker itself is restarted."""

    @classmethod
    def setUpClass(cls):
        loader = SourceFileLoader("coli_cluster_under_test",
                                  str(Path(__file__).resolve().parents[1] / "coli"))
        spec = importlib.util.spec_from_loader(loader.name, loader)
        cls.coli = importlib.util.module_from_spec(spec)
        loader.exec_module(cls.coli)

    def test_worker_registers_again_after_a_coordinator_restart(self):
        coli = self.coli
        servers = [_start()]
        host, port = servers[0].server_address

        class Ticks:
            calls = 0

            def wait(self, _timeout):
                Ticks.calls += 1
                if Ticks.calls == 1:            # the coordinator restarts, empty
                    servers[0].shutdown()
                    servers[0].server_close()
                    servers.append(_start(port))
                return Ticks.calls > 1          # one heartbeat tick, then stop

            def set(self):
                pass

        class Inline:
            def __init__(self, target, **_kw):
                self.target = target

            def start(self):
                self.target()

        a = types.SimpleNamespace(model="/m", cap=24, ebits=8, dbits=8, port=9100,
                                  layers="all", coordinator=f"http://{host}:{port}",
                                  advertise_host="10.0.0.2", node_id="mac-a")
        family = types.SimpleNamespace(descriptor=types.SimpleNamespace(id="glm"))
        try:
            with mock.patch.object(coli, "need_worker_model", return_value="/tmp/engine"), \
                    mock.patch.object(coli, "resolve_model", return_value=family), \
                    mock.patch.object(coli, "env_for", return_value={}), \
                    mock.patch.object(coli, "threading",
                                      types.SimpleNamespace(Event=Ticks, Thread=Inline)), \
                    mock.patch.object(coli.subprocess, "call", return_value=0), \
                    mock.patch("builtins.print"):
                coli.cmd_cluster_worker(a)
            self.assertEqual(servers[-1].registry.expert_endpoints(), ["10.0.0.2:9100"])
        finally:
            servers[-1].shutdown()
            servers[-1].server_close()


if __name__ == "__main__":
    unittest.main()
