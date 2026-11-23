"""c/cluster.py: registration and heartbeat need the shared secret when the coordinator has one.

The coordinator's Host guard stops a web page from re-pointing a name at it, but any host that
can reach the port can send `Host: 127.0.0.1`, register itself as an expert worker, and be handed
to `coli serve` by discovery. With COLI_CLUSTER_TOKEN set on the coordinator and the workers,
register and heartbeat need `Authorization: Bearer <token>`.
"""
import json
import os
import sys
import threading
import unittest
from http.client import HTTPConnection
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cluster
from cluster import ClusterRegistry, ClusterServer, TOKEN_ENV

NODE = {"node_id": "intruder", "host": "198.51.100.7", "port": 9100, "role": "expert"}


def start(token=None):
    server = ClusterServer(("127.0.0.1", 0), ClusterRegistry(), token=token)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def call(server, method, path, body=None, headers=None):
    conn = HTTPConnection(*server.server_address, timeout=5)
    try:
        payload = json.dumps(body) if body is not None else None
        conn.request(method, path, payload, {"Content-Type": "application/json", **(headers or {})})
        response = conn.getresponse()
        return response.status, response.read()
    finally:
        conn.close()


class RegistrationTokenTest(unittest.TestCase):
    def serve(self, token):
        server = start(token)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return server

    def test_without_the_token_nothing_registers(self):
        server = self.serve("s3cret")
        for headers in ({}, {"Authorization": "Bearer wrong"}, {"Authorization": "s3cret"},
                        {"Authorization": "Bearer s3cret "}, {"Authorization": "Bearer s3cré"},
                        {"Authorization": "bearer s3cret"}):
            with self.subTest(headers=headers):
                status, body = call(server, "POST", "/v1/cluster/register", NODE, headers)
                self.assertEqual(status, 401, body)
        self.assertEqual(server.registry.expert_endpoints(), [])

    def test_the_host_guard_is_not_a_substitute(self):
        # what the finding sent: a loopback Host and no credentials
        server = self.serve("s3cret")
        status, _ = call(server, "POST", "/v1/cluster/register", NODE, {"Host": "127.0.0.1"})
        self.assertEqual(status, 401)

    def test_with_the_token_it_registers_and_heartbeats(self):
        server = self.serve("s3cret")
        auth = {"Authorization": "Bearer s3cret"}
        self.assertEqual(call(server, "POST", "/v1/cluster/register", NODE, auth)[0], 200)
        self.assertEqual(server.registry.expert_endpoints(), ["198.51.100.7:9100"])
        self.assertEqual(call(server, "POST", "/v1/cluster/heartbeat", {"node_id": "intruder"}, auth)[0], 200)
        self.assertEqual(call(server, "POST", "/v1/cluster/heartbeat", {"node_id": "intruder"})[0], 401)

    def test_a_good_token_does_not_bypass_the_host_guard(self):
        server = self.serve("s3cret")
        status, _ = call(server, "POST", "/v1/cluster/register", NODE,
                         {"Authorization": "Bearer s3cret", "Host": "attacker.example"})
        self.assertEqual(status, 403)

    def test_reading_the_topology_stays_open(self):
        server = self.serve("s3cret")
        for path in ("/v1/cluster/topology", "/health"):
            self.assertEqual(call(server, "GET", path)[0], 200, path)

    def test_with_no_token_registration_is_open_as_before(self):
        for token in (None, ""):
            with self.subTest(token=token):
                server = self.serve(token)
                self.assertEqual(call(server, "POST", "/v1/cluster/register", NODE)[0], 200)


class ClientTest(unittest.TestCase):
    def setUp(self):
        self.server = start("s3cret")
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.url = "http://%s:%d" % self.server.server_address

    def test_the_client_sends_the_token_from_the_environment(self):
        with mock.patch.dict(os.environ, {TOKEN_ENV: "s3cret"}):
            self.assertEqual(cluster.register(self.url, NODE)["node_id"], "intruder")
            self.assertEqual(cluster.heartbeat(self.url, "intruder")["node_id"], "intruder")
            self.assertEqual(cluster.discover_workers(self.url), ["198.51.100.7:9100"])

    def test_a_worker_without_it_is_told_what_to_set(self):
        env = {k: v for k, v in os.environ.items() if k != TOKEN_ENV}
        with mock.patch.dict(os.environ, env, clear=True):
            with self.assertRaises(OSError) as caught:   # what coli's heartbeat loop already catches
                cluster.register(self.url, NODE)
        self.assertIn(TOKEN_ENV, str(caught.exception))

    def test_a_wrong_token_is_refused_the_same_way(self):
        with mock.patch.dict(os.environ, {TOKEN_ENV: "nope"}):
            with self.assertRaises(OSError):
                cluster.heartbeat(self.url, "intruder")


class ExposureWarningTest(unittest.TestCase):
    def test_open_registration_beyond_this_machine_is_called_out(self):
        for host in ("0.0.0.0", "192.168.1.5", "::"):
            with self.subTest(host=host):
                self.assertIn(TOKEN_ENV, cluster.exposure_warning(host, None))
                self.assertIn(TOKEN_ENV, cluster.exposure_warning(host, ""))

    def test_nothing_to_say_on_loopback_or_with_a_token(self):
        for host in ("127.0.0.1", "localhost", "::1", "[::1]"):
            self.assertIsNone(cluster.exposure_warning(host, None), host)
        self.assertIsNone(cluster.exposure_warning("0.0.0.0", "s3cret"))


if __name__ == "__main__":
    unittest.main()
