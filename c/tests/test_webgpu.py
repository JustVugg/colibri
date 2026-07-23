import socket
import sys
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from webgpu import (WebGPURegistry, _ProxyHandler, _ProxyServer, _recv_exact,
                    _ws_frame, _ws_read_frame, pack_batch, pack_response,
                    parse_batch, parse_response)


class WebGPURuntimeTests(unittest.TestCase):
    def test_batch_and_response_preserve_raw_activation_bytes(self):
        batch = {"layer": 4, "hidden": 2,
                 "items": [{"expert_id": 7, "rows": 1, "activations": b"abcdefgh"}]}
        decoded = parse_batch(pack_batch(batch))
        self.assertEqual(decoded["items"][0]["activations"], b"abcdefgh")
        version, status, count, _ = parse_response(pack_response(decoded, [b"12345678"]))
        self.assertEqual((version, status, count), (1, 0, 1))

    def test_dispatches_batch_to_a_browser_connection(self):
        coordinator, browser = socket.socketpair()
        registry = WebGPURegistry()
        connection = registry.register(coordinator, {"node_id": "browser-a", "expert_ids": ["4:7"]})

        def browser_worker():
            opcode, payload = _ws_read_frame(browser)
            self.assertEqual(opcode, 2)
            incoming = parse_batch(payload)
            browser.sendall(_ws_frame(2, pack_response(incoming, [b"12345678"])))

        worker = threading.Thread(target=browser_worker)
        worker.start()
        batch = {"layer": 4, "hidden": 2,
                 "items": [{"expert_id": 7, "rows": 1, "activations": b"abcdefgh"}]}
        try:
            version, status, count, _ = parse_response(registry.dispatch(pack_batch(batch)))
            self.assertEqual((version, status, count), (1, 0, 1))
            worker.join(timeout=1)
            self.assertFalse(worker.is_alive())
        finally:
            registry.unregister(connection)
            browser.close()

    def test_native_tcp_client_can_use_webgpu_proxy_wire_contract(self):
        registry = WebGPURegistry()
        coordinator, browser = socket.socketpair()
        connection = registry.register(coordinator, {"node_id": "browser-a", "expert_ids": ["4:7"]})
        proxy = _ProxyServer(("127.0.0.1", 0), _ProxyHandler)
        proxy.registry = registry
        thread = threading.Thread(target=proxy.serve_forever, daemon=True)
        thread.start()

        def browser_worker():
            opcode, payload = _ws_read_frame(browser)
            self.assertEqual(opcode, 2)
            incoming = parse_batch(payload)
            browser.sendall(_ws_frame(2, pack_response(incoming, [b"12345678"])))

        worker = threading.Thread(target=browser_worker)
        worker.start()
        native = socket.create_connection(proxy.server_address)
        batch = {"layer": 4, "hidden": 2,
                 "items": [{"expert_id": 7, "rows": 1, "activations": b"abcdefgh"}]}
        try:
            native.sendall(pack_batch(batch))
            header = _recv_exact(native, 20)
            _, status, count = parse_response(header)[0:3]
            payload = _recv_exact(native, 8 + 8)
            self.assertEqual((status, count), (0, 1))
            self.assertEqual(parse_response(header + payload)[3][-8:], b"12345678")
            worker.join(timeout=1)
            self.assertFalse(worker.is_alive())
        finally:
            native.close()
            proxy.shutdown(); proxy.server_close()
            registry.unregister(connection)
            browser.close()


if __name__ == "__main__":
    unittest.main()
