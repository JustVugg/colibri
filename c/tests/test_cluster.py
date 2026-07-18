import json
import socket
import threading
import unittest
from http.client import HTTPConnection

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cluster import (ClusterRegistry, ClusterServer, PROTOCOL_VERSION,
                     pack_expert_batch, pack_expert_response,
                     parse_expert_batch, parse_expert_response,
                     _ws_frame, _ws_read_frame)


class ClusterRegistryTests(unittest.TestCase):
    def test_expert_wire_round_trip_preserves_activation_bytes(self):
        batch = {"layer": 4, "hidden": 2, "items": [
            {"expert_id": 7, "rows": 1, "activations": b"abcdefgh"},
        ]}
        decoded = parse_expert_batch(pack_expert_batch(batch))
        self.assertEqual(decoded["layer"], 4)
        self.assertEqual(decoded["items"][0]["activations"], b"abcdefgh")
        response = parse_expert_response(pack_expert_response(decoded, [b"12345678"]))
        self.assertEqual((response["status"], response["count"]), (0, 1))

    def test_registers_and_discovers_expert_nodes(self):
        registry = ClusterRegistry()
        registry.register({"node_id": "mac-a", "host": "10.0.0.2", "port": 9100,
                           "role": "expert", "layers": "0-37"})
        registry.register({"node_id": "mac-b", "host": "10.0.0.3", "port": 9101,
                           "role": "dense", "layers": "38-75"})
        self.assertEqual(registry.expert_endpoints(), ["10.0.0.2:9100"])
        self.assertEqual(registry.snapshot()["protocol_version"], PROTOCOL_VERSION)

    def test_accepts_browser_webgpu_capability_record(self):
        registry = ClusterRegistry()
        registry.register({"node_id": "iphone-a", "host": "browser", "port": 0,
                           "role": "webgpu", "device_type": "iPhone", "precision": "f32",
                           "expert_ids": ["4:7"]})
        self.assertEqual(registry.snapshot()["nodes"][0]["role"], "webgpu")

    def test_dispatches_native_batch_over_websocket_connection(self):
        coordinator, browser = socket.socketpair()
        connection = None
        try:
            registry = ClusterRegistry()
            connection = registry.register_webgpu_connection(
                coordinator,
                {"node_id": "browser-a", "expert_ids": ["4:7"]},
            )

            def browser_worker():
                opcode, payload = _ws_read_frame(browser)
                self.assertEqual(opcode, 2)
                batch = parse_expert_batch(payload)
                response = pack_expert_response(batch, [b"12345678"])
                browser.sendall(_ws_frame(2, response))

            worker = threading.Thread(target=browser_worker)
            worker.start()
            batch = {"layer": 4, "hidden": 2, "items": [
                {"expert_id": 7, "rows": 1, "activations": b"abcdefgh"},
            ]}
            response = parse_expert_response(registry.dispatch_webgpu(pack_expert_batch(batch)))
            self.assertEqual(response["status"], 0)
            self.assertEqual(response["count"], 1)
            worker.join(timeout=1)
            self.assertFalse(worker.is_alive())
        finally:
            if connection is not None:
                registry.unregister_webgpu(connection)
            browser.close()

    def test_http_topology_and_registration(self):
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
