"""Worker leases measure elapsed time while public timestamps stay wall time."""
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from cluster import ClusterRegistry


class ClusterLeaseClockTest(unittest.TestCase):
    def test_forward_wall_clock_change_does_not_expire_live_worker(self):
        registry = ClusterRegistry(stale_after=30)
        with patch('cluster.time', SimpleNamespace(time=lambda: 1000, monotonic=lambda: 10)):
            record = registry.register({'node_id': 'worker', 'host': 'localhost', 'port': 9100, 'role': 'expert'})
        self.assertEqual(record['last_seen'], 1000)
        with patch('cluster.time', SimpleNamespace(time=lambda: 4600, monotonic=lambda: 11)):
            self.assertEqual(registry.expert_endpoints(), ['localhost:9100'])

    def test_backward_wall_clock_change_does_not_keep_dead_worker(self):
        registry = ClusterRegistry(stale_after=30)
        with patch('cluster.time', SimpleNamespace(time=lambda: 1000, monotonic=lambda: 10)):
            registry.register({'node_id': 'worker', 'host': 'localhost', 'port': 9100, 'role': 'expert'})
        with patch('cluster.time', SimpleNamespace(time=lambda: 100, monotonic=lambda: 41)):
            self.assertEqual(registry.expert_endpoints(), [])

    def test_lease_lookup_uses_registry_key_not_returned_record_id(self):
        registry = ClusterRegistry(stale_after=30)
        record = registry.register({'node_id': 'worker', 'host': 'localhost', 'port': 9100, 'role': 'expert'})
        # register has historically returned the record itself. Its public
        # fields must not become keys into the private elapsed-time map.
        record['node_id'] = 'display-id'
        self.assertEqual(registry.expert_endpoints(), ['localhost:9100'])

    def test_heartbeat_renews_lease_and_keeps_public_wall_timestamp(self):
        registry = ClusterRegistry(stale_after=30)
        with patch('cluster.time', SimpleNamespace(time=lambda: 1000, monotonic=lambda: 10)):
            registry.register({'node_id': 1, 'host': 'localhost', 'port': 9100, 'role': 'expert'})
        with patch('cluster.time', SimpleNamespace(time=lambda: 500, monotonic=lambda: 39)):
            self.assertEqual(registry.heartbeat(1)['last_seen'], 500)
        with patch('cluster.time', SimpleNamespace(time=lambda: 800, monotonic=lambda: 68)):
            self.assertEqual(registry.expert_endpoints(), ['localhost:9100'])
        with patch('cluster.time', SimpleNamespace(time=lambda: 800, monotonic=lambda: 70)):
            self.assertEqual(registry.expert_endpoints(), [])


if __name__ == '__main__':
    unittest.main()
