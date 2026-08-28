import json
import sys
import threading
import unittest
from http.client import HTTPConnection
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
