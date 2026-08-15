"""Durable lifecycle integration tests for cgroup-v2 managed engines."""

import copy
import os
import sys
import unittest
from unittest import mock
from pathlib import Path


C_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(C_DIR))

from ramdisk_support import lifecycle  # noqa: E402
from ramdisk_support import processes  # noqa: E402
from ramdisk_support import state  # noqa: E402
from ramdisk_support import supervision  # noqa: E402
from ramdisk_support.common import RamdiskError  # noqa: E402


def containment(index=1):
    return {
        "version": 1,
        "mode": "cgroup-v2",
        "relative_path": "colibri/deployment/operation-%d" % index,
        "device": 100,
        "inode": 200 + index,
    }


class FakeProcess:
    def __init__(self, pid):
        self.pid = pid
        self.returncode = None

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.returncode = 125
        return self.returncode

    def kill(self):
        self.returncode = -9


class FakeGate:
    def __init__(self, pid):
        self.process = FakeProcess(pid)
        self.release_fd = 91
        self.pidfd = None
        self.released = False


class RecordingSupervisor:
    def __init__(self, *, missing=False):
        self.events = []
        self.missing = missing
        self.next_pid = 4100
        self.gates = {}

    def create_leaf(self, deployment_id, operation_id):
        self.events.append(("create", deployment_id, operation_id))
        return containment(self.next_pid - 4099)

    def spawn_gate(self, command, *, environment, **kwargs):
        self.events.append(("spawn", list(command), dict(environment)))
        return FakeGate(self.next_pid)

    def attach_gate(self, gate, authority):
        self.events.append(("attach", gate.process.pid, copy.deepcopy(authority)))
        gate.pidfd = 92
        self.gates[authority["relative_path"]] = gate

    def release_gate(self, gate, authority):
        self.events.append(("release", gate.process.pid, copy.deepcopy(authority)))
        gate.released = True

    def abort_gate(self, gate):
        self.events.append(("abort", gate.process.pid))
        gate.process.returncode = 125
        gate.release_fd = None
        gate.pidfd = None

    def close_gate(self, gate):
        self.events.append(("close", gate.process.pid))
        gate.pidfd = None

    def members(self, authority):
        self.events.append(("preflight", authority["relative_path"]))
        if self.missing:
            raise supervision.ContainmentMissing("missing cgroup")
        return [4100]

    def terminate(self, authority, **kwargs):
        self.events.append(("terminate", authority["relative_path"]))
        gate = self.gates.get(authority["relative_path"])
        if gate is not None:
            gate.process.returncode = -15
        return {"status": "absent"}

    def prove_absence(self, authority):
        self.events.append(("absence", authority["relative_path"]))
        if self.missing:
            raise supervision.ContainmentMissing("missing cgroup")
        return True

    def remove_empty(self, authority):
        self.events.append(("remove", authority["relative_path"]))
        if self.missing:
            raise supervision.ContainmentMissing("missing cgroup")
        return True


class DurableGatePhaseTest(unittest.TestCase):
    def test_gate_phase_is_saved_before_each_irreversible_transition(self):
        supervisor = RecordingSupervisor()
        pending = {
            "operation_id": "start:" + "a" * 32,
            "containment": containment(),
            "containment_phase": "cgroup-created",
        }
        manifest = {"pending_launches": [pending]}
        context = {"pending_entry": pending}
        snapshots = []

        gate = lifecycle._launch_contained_gate(
            ["coli", "serve"],
            {"COLI_MANAGED_NONCE": "n"},
            context,
            manifest=manifest,
            supervisor=supervisor,
            save_manifest=lambda value: snapshots.append(copy.deepcopy(value)),
        )

        self.assertIs(gate, context["gate"])
        self.assertEqual(
            [item["pending_launches"][0]["containment_phase"] for item in snapshots],
            ["gate-spawned", "attached-verified", "gate-released"],
        )
        self.assertEqual(pending["pid"], 4100)
        self.assertEqual(
            [event[0] for event in supervisor.events],
            ["spawn", "attach", "release"],
        )


class ContainmentStateValidationTest(unittest.TestCase):
    def test_pending_phase_requires_pid_only_after_spawn(self):
        initial = {
            "containment": containment(),
            "containment_phase": "cgroup-created",
        }
        state._validate_supervised_containment(initial, pending=True)

        spawned = dict(initial, containment_phase="gate-spawned", pid=4100)
        state._validate_supervised_containment(spawned, pending=True)

        with self.assertRaisesRegex(RamdiskError, "gate identity"):
            state._validate_supervised_containment(
                dict(initial, pid=4100),
                pending=True,
            )
        with self.assertRaisesRegex(RamdiskError, "gate identity"):
            state._validate_supervised_containment(
                dict(initial, containment_phase="attached-verified"),
                pending=True,
            )

    def test_removed_marker_requires_prior_durable_authorization(self):
        with self.assertRaisesRegex(RamdiskError, "removal state"):
            state._validate_supervised_containment(
                {
                    "containment": containment(),
                    "containment_removed_at": "2026-08-08T12:00:00Z",
                },
                pending=False,
            )


class DurableRemovalTest(unittest.TestCase):
    def test_removal_authority_is_durable_before_rmdir(self):
        supervisor = RecordingSupervisor()
        record = {"containment": containment()}
        manifest = {"processes": [record]}
        snapshots = []

        lifecycle._retire_containment(
            record,
            manifest=manifest,
            supervisor=supervisor,
            save_manifest=lambda value: snapshots.append(copy.deepcopy(value)),
            terminate=True,
        )

        self.assertIn("containment_removal_authorized_at", snapshots[0]["processes"][0])
        self.assertNotIn("containment_removed_at", snapshots[0]["processes"][0])
        self.assertIn("containment_removed_at", snapshots[1]["processes"][0])
        self.assertEqual(
            [event[0] for event in supervisor.events],
            ["terminate", "absence", "remove"],
        )

    def test_missing_leaf_is_inconclusive_without_durable_removal_authority(self):
        supervisor = RecordingSupervisor(missing=True)
        record = {"containment": containment()}

        with self.assertRaises(supervision.ContainmentMissing):
            lifecycle._retire_containment(
                record,
                manifest={"processes": [record]},
                supervisor=supervisor,
                save_manifest=mock.Mock(),
                terminate=False,
            )
        self.assertNotIn("containment_removed_at", record)

    def test_missing_leaf_can_complete_an_already_durable_removal(self):
        supervisor = RecordingSupervisor(missing=True)
        record = {
            "containment": containment(),
            "containment_removal_authorized_at": "2026-08-08T12:00:00Z",
        }
        saved = []

        lifecycle._retire_containment(
            record,
            manifest={"processes": [record]},
            supervisor=supervisor,
            save_manifest=lambda value: saved.append(copy.deepcopy(value)),
            terminate=False,
        )

        self.assertIn("containment_removed_at", record)
        self.assertEqual(len(saved), 1)


class ContainedRollbackTest(unittest.TestCase):
    def _context(self, *, released):
        gate = FakeGate(4100)
        gate.released = released
        pending = {
            "operation_id": "start:" + "a" * 32,
            "state_dir": "/state/engine",
            "usage_baseline": {},
            "usage_merge_id": "a" * 32,
            "containment": containment(),
            "containment_phase": (
                "gate-released" if released else "gate-spawned"
            ),
            "pid": 4100,
        }
        return {
            "gate": gate,
            "state_dir": pending["state_dir"],
            "usage_merge_id": pending["usage_merge_id"],
            "pending_entry": pending,
            "record": None,
        }

    def test_released_child_rolls_back_through_cgroup_then_removes_pending(self):
        supervisor = RecordingSupervisor()
        context = self._context(released=True)
        supervisor.gates[
            context["pending_entry"]["containment"]["relative_path"]
        ] = context["gate"]
        manifest = {"pending_launches": [context["pending_entry"]], "processes": []}

        failures = lifecycle._rollback_contained_launches(
            manifest,
            [context],
            supervisor=supervisor,
            save_manifest=mock.Mock(),
            merge_usage=mock.Mock(),
            canonical_usage="/model/.coli_usage",
            plan={},
            forget_managed_child=mock.Mock(),
        )

        self.assertEqual(failures, [])
        self.assertEqual(manifest["pending_launches"], [])
        self.assertIn("containment_removed_at", context["pending_entry"])
        self.assertIn("terminate", [event[0] for event in supervisor.events])

    def test_blocked_child_is_aborted_before_absence_and_never_signaled_by_pid(self):
        supervisor = RecordingSupervisor()
        context = self._context(released=False)
        manifest = {"pending_launches": [context["pending_entry"]], "processes": []}

        failures = lifecycle._rollback_contained_launches(
            manifest,
            [context],
            supervisor=supervisor,
            save_manifest=mock.Mock(),
            merge_usage=mock.Mock(),
            canonical_usage="/model/.coli_usage",
            plan={},
            forget_managed_child=mock.Mock(),
        )

        self.assertEqual(failures, [])
        event_names = [event[0] for event in supervisor.events]
        self.assertLess(event_names.index("abort"), event_names.index("absence"))
        self.assertNotIn("terminate", event_names)


class ContainedStopTest(unittest.TestCase):
    def _record(self, index):
        return {
            "pid": 4099 + index,
            "pgid": 4099 + index,
            "port": 7999 + index,
            "node": index,
            "state_dir": "/state/%d" % index,
            "usage_baseline": {},
            "usage_merge_id": ("%x" % index) * 32,
            "containment": containment(index),
        }

    def test_all_containments_are_preflighted_before_first_signal(self):
        supervisor = RecordingSupervisor()
        records = [self._record(1), self._record(2)]
        manifest = {
            "process_supervision_version": 1,
            "state": "running",
            "plan": {
                "model": {"path": "/model"},
                "mounts": [
                    {"path": "/weights/1"},
                    {"path": "/weights/2"},
                ],
            },
            "mounts": [
                {"path": "/weights/1", "ownership": "managed"},
                {"path": "/weights/2", "ownership": "managed"},
            ],
            "processes": records,
            "pending_launches": [],
        }

        result = lifecycle.stop(
            load_manifest=lambda required: manifest,
            discover_managed_launches=mock.Mock(),
            process_matches=mock.Mock(),
            process_group_members=mock.Mock(),
            group_alive=mock.Mock(),
            managed_child_liveness=mock.Mock(),
            save_manifest=mock.Mock(),
            terminate_verified_group=mock.Mock(),
            merge_usage=mock.Mock(),
            bind_usage_transaction=lambda record, **kwargs: record["usage_merge_id"],
            containment_supervisor=supervisor,
        )

        first_terminate = next(
            index for index, event in enumerate(supervisor.events)
            if event[0] == "terminate"
        )
        self.assertEqual(
            [event[0] for event in supervisor.events[:first_terminate]],
            ["preflight", "preflight"],
        )
        self.assertEqual(result["state"], "stopped")
        self.assertTrue(all("containment_removed_at" in item for item in records))

    def test_missing_containment_refuses_before_any_signal(self):
        supervisor = RecordingSupervisor(missing=True)
        record = self._record(1)
        manifest = {
            "process_supervision_version": 1,
            "state": "running",
            "plan": {"model": {"path": "/model"}, "mounts": []},
            "mounts": [],
            "processes": [record],
            "pending_launches": [],
        }

        with self.assertRaisesRegex(RamdiskError, "containment preflight"):
            lifecycle.stop(
                load_manifest=lambda required: manifest,
                discover_managed_launches=mock.Mock(),
                process_matches=mock.Mock(),
                process_group_members=mock.Mock(),
                group_alive=mock.Mock(),
                managed_child_liveness=mock.Mock(),
                save_manifest=mock.Mock(),
                terminate_verified_group=mock.Mock(),
                merge_usage=mock.Mock(),
                bind_usage_transaction=mock.Mock(),
                containment_supervisor=supervisor,
            )
        self.assertFalse(any(event[0] == "terminate" for event in supervisor.events))


class ContainedProcessRoutingTest(unittest.TestCase):
    def test_removed_containment_is_authoritative_absence(self):
        record = {
            "pid": 4100,
            "containment": containment(),
            "containment_removal_authorized_at": "2026-08-08T12:00:00Z",
            "containment_removed_at": "2026-08-08T12:00:01Z",
        }
        proc_identity = mock.Mock()

        self.assertEqual(
            processes._process_matches(
                record,
                proc_identity=proc_identity,
                containment_supervisor=mock.Mock(),
            ),
            (False, "not-running", None),
        )
        proc_identity.assert_not_called()

    def test_unverified_live_containment_fails_closed_before_procfs(self):
        record = {"pid": 4100, "containment": containment()}
        supervisor = mock.Mock()
        supervisor.verify_record.return_value = (False, "leader not a member")
        supervisor.liveness.return_value = True
        proc_identity = mock.Mock()

        matches, reason, actual = processes._process_matches(
            record,
            proc_identity=proc_identity,
            containment_supervisor=supervisor,
        )

        self.assertFalse(matches)
        self.assertEqual(reason, "unverified-containment")
        self.assertEqual(actual["reason"], "leader not a member")
        proc_identity.assert_not_called()

    def test_termination_uses_cgroup_membership_not_process_group_ops(self):
        record = {"pid": 4100, "containment": containment()}
        supervisor = mock.Mock()
        supervisor.terminate.return_value = {"status": "absent"}
        ops = mock.Mock()

        self.assertIsNone(
            processes._terminate_verified_group(
                record,
                containment_supervisor=supervisor,
                ops=ops,
            )
        )
        supervisor.terminate.assert_called_once_with(
            record["containment"],
            term_seconds=10.0,
            kill_seconds=3.0,
        )
        ops.signal_verified_process_group.assert_not_called()


class ContainedStatusTest(unittest.TestCase):
    def test_status_uses_durable_removed_marker_without_pid_fallback(self):
        record = {
            "pid": 4100,
            "port": 8000,
            "node": None,
            "state_dir": "/state/engine",
            "stopped_at": "2026-08-08T12:00:02Z",
            "containment": containment(),
            "containment_removal_authorized_at": "2026-08-08T12:00:00Z",
            "containment_removed_at": "2026-08-08T12:00:01Z",
        }
        manifest = {
            "process_supervision_version": 1,
            "state": "stopped",
            "plan": {"mode": "fit", "topology": "interleaved", "mounts": []},
            "mounts": [],
            "processes": [record],
            "pending_launches": [],
            "ports": [8000],
        }
        process_matches = mock.Mock()
        child_liveness = mock.Mock(return_value=False)
        supervisor = mock.Mock()

        report = lifecycle.status(
            deep=False,
            load_manifest=lambda required: manifest,
            manifest_path=lambda: "/state/manifest.json",
            source_still_matches=mock.Mock(),
            mount_at=mock.Mock(),
            validate_mount=mock.Mock(),
            validate_namespace=mock.Mock(),
            process_matches=process_matches,
            managed_child_liveness=child_liveness,
            containment_supervisor=supervisor,
        )

        self.assertFalse(report["processes"][0]["running"])
        self.assertEqual(report["processes"][0]["reason"], "stopped")
        process_matches.assert_not_called()
        child_liveness.assert_called_once_with(4100)
        supervisor.liveness.assert_not_called()

    def test_status_reports_exact_pending_gate_phase(self):
        pending = {
            "operation_id": "start:" + "a" * 32,
            "port": 8000,
            "node": None,
            "state_dir": "/state/engine",
            "containment": containment(),
            "containment_phase": "attached-verified",
            "pid": 4100,
        }
        manifest = {
            "process_supervision_version": 1,
            "state": "starting",
            "plan": {"mode": "fit", "topology": "interleaved", "mounts": []},
            "mounts": [],
            "processes": [],
            "pending_launches": [pending],
            "ports": [],
        }

        report = lifecycle.status(
            deep=False,
            load_manifest=lambda required: manifest,
            manifest_path=lambda: "/state/manifest.json",
            source_still_matches=mock.Mock(),
            mount_at=mock.Mock(),
            validate_mount=mock.Mock(),
            validate_namespace=mock.Mock(),
            process_matches=mock.Mock(),
            managed_child_liveness=mock.Mock(),
            containment_supervisor=mock.Mock(),
        )

        self.assertEqual(
            report["recovery"]["pending_launches"][0]["state"],
            "attached-verified",
        )


if __name__ == "__main__":
    unittest.main()
