import dataclasses
import sys
import unittest
from pathlib import Path


C_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(C_DIR))

from ramdisk_ui import (  # noqa: E402
    ActionPolicy,
    DeploymentHealth,
    HealthLevel,
    PlacementContract,
    ReviewAction,
    ReviewIdentity,
)


def placement_plan(topology="interleaved", nodes=4):
    if topology == "interleaved":
        mounts = [{"node": None, "path": "/mnt/colibri-ram/shared"}]
        replicas = 1
    else:
        mounts = [
            {"node": node, "path": "/mnt/colibri-ram/node%d" % node}
            for node in range(nodes)
        ]
        replicas = nodes
    staged = 372 * (1 << 30)
    return {
        "topology": topology,
        "mode": "full",
        "hardware": {
            "online_nodes": list(range(nodes)),
            # Slot-like metadata must not be interpreted as a controllable DIMM
            # placement policy. The current contract exposes NUMA only.
            "nodes": [{"id": node, "slots": ["A%d" % node]} for node in range(nodes)],
        },
        "staging": {
            "replica_count": replicas,
            "staged_bytes": staged,
            "total_staged_bytes": staged * replicas,
        },
        "mounts": mounts,
        "blockers": [],
    }


class PlacementContractTest(unittest.TestCase):
    def test_four_node_shared_plan_is_one_copy_and_one_engine(self):
        contract = PlacementContract.from_plan(placement_plan(), base_port=8000)

        self.assertTrue(contract.is_shared)
        self.assertFalse(contract.is_replication)
        self.assertEqual(contract.copy_count, 1)
        self.assertEqual(contract.engine_count, 1)
        self.assertEqual(contract.numa_nodes, (0, 1, 2, 3))
        self.assertEqual(contract.ports, (8000,))
        self.assertEqual(contract.total_staged_bytes, contract.staged_bytes_per_copy)

    def test_four_node_replica_plan_exposes_every_multiplier(self):
        contract = PlacementContract.from_plan(
            placement_plan("per-node"), base_port=8100
        )

        self.assertTrue(contract.is_replication)
        self.assertEqual(contract.copy_count, 4)
        self.assertEqual(contract.engine_count, 4)
        self.assertEqual(contract.ports, (8100, 8101, 8102, 8103))
        self.assertEqual(
            contract.total_staged_bytes, contract.staged_bytes_per_copy * 4
        )
        self.assertEqual(
            contract.mount_paths,
            tuple("/mnt/colibri-ram/node%d" % node for node in range(4)),
        )

    def test_projection_is_immutable_and_does_not_infer_dimm_groups(self):
        contract = PlacementContract.from_plan(placement_plan())

        with self.assertRaises(dataclasses.FrozenInstanceError):
            contract.copy_count = 4
        self.assertEqual(
            set(dataclasses.asdict(contract)),
            {
                "topology",
                "mode",
                "copy_count",
                "engine_count",
                "staged_bytes_per_copy",
                "total_staged_bytes",
                "numa_nodes",
                "ports",
                "mount_paths",
            },
        )

    def test_projection_derives_copy_cost_from_actual_topology(self):
        plan = placement_plan("per-node")
        plan["staging"]["replica_count"] = 1
        plan["staging"]["total_staged_bytes"] = plan["staging"]["staged_bytes"]

        contract = PlacementContract.from_plan(plan)

        self.assertEqual(contract.copy_count, 4)
        self.assertEqual(
            contract.total_staged_bytes,
            contract.staged_bytes_per_copy * contract.copy_count,
        )


class DeploymentHealthTest(unittest.TestCase):
    def setUp(self):
        self.plan = placement_plan()
        self.mount = {
            "verified": True,
            "namespace_verified": True,
        }

    def test_deeply_verified_deployment_is_distinct_from_fast_check(self):
        verified = DeploymentHealth.from_report(
            self.plan,
            {
                "state": "ready",
                "mounts": [self.mount],
                "processes": [],
                "deep_validation": True,
                "source_fingerprint_verified": True,
            },
        )
        fast = DeploymentHealth.from_report(
            self.plan,
            {
                "state": "ready",
                "mounts": [{"verified": True, "namespace_verified": None}],
                "processes": [],
                "deep_validation": False,
                "source_fingerprint_verified": None,
            },
        )

        self.assertEqual(verified.level, HealthLevel.VERIFIED)
        self.assertTrue(verified.verified)
        self.assertIs(verified.source_fingerprint_verified, True)
        self.assertTrue(verified.namespaces_verified)
        self.assertEqual(fast.level, HealthLevel.FAST_CHECK)
        self.assertTrue(fast.fast_check_ok)
        self.assertFalse(fast.verified)
        self.assertIsNone(fast.source_fingerprint_verified)
        self.assertFalse(fast.namespaces_verified)

    def test_running_state_requires_every_expected_verified_process(self):
        health = DeploymentHealth.from_report(
            self.plan,
            {
                "state": "running",
                "mounts": [self.mount],
                "processes": [{"running": False, "verified": False}],
                "deep_validation": True,
                "source_fingerprint_verified": True,
            },
        )

        self.assertEqual(health.level, HealthLevel.NEEDS_ATTENTION)
        self.assertFalse(health.processes_healthy)
        self.assertFalse(health.fast_check_ok)

    def test_ready_state_rejects_foreign_process_records(self):
        health = DeploymentHealth.from_report(
            self.plan,
            {
                "state": "ready",
                "mounts": [self.mount],
                "processes": [
                    {
                        "running": False,
                        "verified": False,
                        "reason": "nonce-mismatch",
                    }
                ],
                "deep_validation": False,
                "source_fingerprint_verified": None,
            },
        )

        self.assertEqual(health.level, HealthLevel.NEEDS_ATTENTION)
        self.assertFalse(health.processes_healthy)

    def test_stopped_state_accepts_benign_stopped_process_records(self):
        health = DeploymentHealth.from_report(
            self.plan,
            {
                "state": "stopped",
                "mounts": [self.mount],
                "processes": [
                    {"running": False, "verified": False, "reason": "stopped"}
                ],
                "deep_validation": False,
                "source_fingerprint_verified": None,
            },
        )

        self.assertEqual(health.level, HealthLevel.FAST_CHECK)
        self.assertTrue(health.processes_healthy)

    def test_replica_health_requires_every_expected_mount(self):
        health = DeploymentHealth.from_report(
            placement_plan("per-node"),
            {
                "state": "ready",
                "mounts": [self.mount] * 3,
                "processes": [],
                "deep_validation": True,
                "source_fingerprint_verified": True,
            },
        )

        self.assertEqual(health.level, HealthLevel.NEEDS_ATTENTION)
        self.assertFalse(health.mounts_healthy)


class ActionPolicyTest(unittest.TestCase):
    def test_absent_ready_plan_allows_only_prepare_and_settings(self):
        policy = ActionPolicy.from_state(
            placement_plan(), {"present": False, "state": "absent"}
        )

        self.assertTrue(policy.prepare.enabled)
        self.assertTrue(policy.edit_weights.enabled)
        self.assertTrue(policy.edit_base_port.enabled)
        self.assertFalse(policy.start.enabled)
        self.assertFalse(policy.stop.enabled)
        self.assertFalse(policy.destroy.enabled)
        self.assertFalse(policy.benchmark.enabled)

    def test_active_state_matrix_keeps_weight_settings_locked(self):
        expected = {
            "ready": (True, False, True, True, True),
            "stopped": (True, False, True, True, True),
            "running": (False, True, False, False, False),
            "starting": (False, True, False, False, False),
            "error": (False, False, True, False, False),
        }
        for state, values in expected.items():
            with self.subTest(state=state):
                policy = ActionPolicy.from_state(
                    placement_plan(), {"present": True, "state": state}
                )
                self.assertEqual(
                    (
                        policy.start.enabled,
                        policy.stop.enabled,
                        policy.destroy.enabled,
                        policy.benchmark.enabled,
                        policy.edit_base_port.enabled,
                    ),
                    values,
                )
                self.assertFalse(policy.prepare.enabled)
                self.assertFalse(policy.edit_weights.enabled)

    def test_error_state_only_offers_stop_when_process_cleanup_is_pending(self):
        no_process = ActionPolicy.from_state(
            placement_plan(), {"present": True, "state": "error", "processes": []}
        )
        pending_process = ActionPolicy.from_state(
            placement_plan(),
            {"present": True, "state": "error", "processes": [{"pid": 123}]},
        )
        already_stopped = ActionPolicy.from_state(
            placement_plan(),
            {
                "present": True,
                "state": "error",
                "processes": [{"pid": 123, "running": False, "reason": "stopped"}],
            },
        )

        self.assertFalse(no_process.stop.enabled)
        self.assertIn("Destroy", no_process.stop.reason)
        self.assertFalse(no_process.edit_base_port.enabled)
        self.assertIn("Destroy", no_process.edit_base_port.reason)
        self.assertTrue(pending_process.stop.enabled)
        self.assertEqual(pending_process.stop.reason, "")
        self.assertFalse(pending_process.destroy.enabled)
        self.assertIn("Stop every managed engine", pending_process.destroy.reason)
        self.assertIn("Stop managed engines", pending_process.edit_base_port.reason)
        self.assertFalse(already_stopped.stop.enabled)
        self.assertTrue(already_stopped.destroy.enabled)
        self.assertIn("Destroy", already_stopped.edit_base_port.reason)

    def test_retained_process_recovery_offers_only_stop_reconciliation(self):
        policy = ActionPolicy.from_state(
            placement_plan(),
            {
                "present": True,
                "state": "error",
                "processes": [],
                "recovery": {
                    "retained_processes": [{"pid": 123, "pgid": 123}],
                    "pending_launches": [],
                },
            },
        )

        self.assertTrue(policy.stop.enabled)
        self.assertEqual(policy.stop.reason, "")
        self.assertFalse(policy.destroy.enabled)
        self.assertIn("reconcile", policy.destroy.reason)
        self.assertFalse(policy.edit_base_port.enabled)
        self.assertIn("Stop managed engines", policy.edit_base_port.reason)

    def test_outcome_unknown_launch_disables_stop_and_destroy(self):
        policy = ActionPolicy.from_state(
            placement_plan(),
            {
                "present": True,
                "state": "error",
                "processes": [],
                "recovery": {
                    "retained_processes": [],
                    "pending_launches": [
                        {"port": 8000, "state": "outcome-unknown"}
                    ],
                },
            },
        )

        self.assertFalse(policy.stop.enabled)
        self.assertFalse(policy.destroy.enabled)
        self.assertIn("outcome is unknown", policy.stop.reason)
        self.assertIn("outcome is unknown", policy.destroy.reason)
        self.assertFalse(policy.edit_base_port.enabled)
        self.assertIn("outcome-unknown", policy.edit_base_port.reason)

    def test_unknown_or_blocked_state_has_actionable_reasons(self):
        unknown = ActionPolicy.from_state(None, None)
        blocked_plan = placement_plan()
        blocked_plan["blockers"] = ["not enough memory"]
        blocked = ActionPolicy.from_state(
            blocked_plan, {"present": False, "state": "absent"}
        )

        self.assertFalse(unknown.prepare.enabled)
        self.assertIn("blocked", unknown.prepare.reason.lower())
        self.assertFalse(blocked.prepare.enabled)
        self.assertIn("NOT READY", blocked.prepare.reason)

    def test_enabled_actions_have_no_error_reason_and_port_lock_is_specific(self):
        ready = ActionPolicy.from_state(
            placement_plan(), {"present": True, "state": "ready"}
        )
        running = ActionPolicy.from_state(
            placement_plan(), {"present": True, "state": "running"}
        )

        self.assertEqual(ready.start.reason, "")
        self.assertEqual(ready.destroy.reason, "")
        self.assertEqual(ready.benchmark.reason, "")
        self.assertEqual(ready.edit_base_port.reason, "")
        self.assertFalse(running.destroy.enabled)
        self.assertIn("Stop every managed engine", running.destroy.reason)
        self.assertNotIn("Stop", ready.edit_weights.reason)
        self.assertIn("Destroy", ready.edit_weights.reason)
        self.assertIn("Stop managed engines", running.edit_base_port.reason)
        self.assertIn("Stop and Destroy", running.edit_weights.reason)
        self.assertIn("Destroy", running.edit_weights.reason)


class ReviewIdentityTest(unittest.TestCase):
    def test_prepare_review_includes_endpoint_facts_beyond_plan_token(self):
        plan = placement_plan()
        reviewed = ReviewIdentity.for_prepare("token", plan, base_port=8000)
        changed_endpoint = ReviewIdentity.for_prepare("token", plan, base_port=9000)

        self.assertEqual(reviewed.action, ReviewAction.PREPARE)
        self.assertEqual(reviewed.token, "token")
        self.assertNotEqual(reviewed, changed_endpoint)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            reviewed.token = "replacement"

    def test_destroy_review_includes_exact_manifest_mount_paths(self):
        plan = placement_plan()
        reviewed = ReviewIdentity.for_destroy(
            "token", plan, [{"path": "/mnt/colibri-ram/shared"}]
        )
        replacement = ReviewIdentity.for_destroy(
            "token", plan, [{"path": "/mnt/colibri-ram/replacement"}]
        )

        self.assertEqual(reviewed.action, ReviewAction.DESTROY)
        self.assertNotEqual(reviewed, replacement)

    def test_destroy_review_includes_the_persisted_endpoint(self):
        reviewed = ReviewIdentity.for_destroy(
            "token",
            placement_plan(),
            [{"path": "/mnt/colibri-ram/shared"}],
            base_port=9123,
        )

        self.assertEqual(reviewed.placement.ports, (9123,))


if __name__ == "__main__":
    unittest.main()
