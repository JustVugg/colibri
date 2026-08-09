"""Deterministic cgroup-v2 containment and gated-exec tests."""

import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock


C_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(C_DIR))

from ramdisk_support.common import RamdiskError  # noqa: E402
from ramdisk_support import supervision  # noqa: E402

if __package__:
    from .platform_test_support import (  # noqa: E402
        requires_native_dirfd,
        requires_posix_pass_fds,
    )
else:
    from platform_test_support import (  # noqa: E402
        requires_native_dirfd,
        requires_posix_pass_fds,
    )


class CgroupFixture:
    def __init__(self, root):
        self.root = Path(root)
        (self.root / "cgroup.controllers").write_text("memory pids\n")

    def supervisor(self, **kwargs):
        return supervision.CgroupSupervisor(root=str(self.root), **kwargs)

    def materialize_leaf(self, containment, *, pids=(), populated=None):
        leaf = self.root / containment["relative_path"]
        leaf.mkdir(parents=True, exist_ok=True)
        (leaf / "cgroup.procs").write_text(
            "".join("%d\n" % pid for pid in pids)
        )
        if populated is None:
            populated = bool(pids)
        (leaf / "cgroup.events").write_text(
            "populated %d\nfrozen 0\n" % int(populated)
        )
        info = leaf.stat()
        containment.update(device=info.st_dev, inode=info.st_ino)
        return leaf


class StableCgroupIdentityTest(unittest.TestCase):
    def test_delegated_root_must_be_absolute(self):
        with self.assertRaisesRegex(RamdiskError, "absolute path"):
            supervision.CgroupSupervisor(root="relative/delegation")

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.fixture = CgroupFixture(self.temporary.name)
        self.supervisor = self.fixture.supervisor()

    @requires_native_dirfd
    def test_create_leaf_returns_canonical_stable_identity(self):
        expected_relative = self.supervisor.relative_leaf(
            "deployment-a",
            "start:operation-a",
        )
        self.assertIsNone(
            self.supervisor.reconcile_leaf_intent(
                "deployment-a",
                "start:operation-a",
            )
        )
        containment = self.supervisor.create_leaf(
            "deployment-a",
            "start:operation-a",
        )

        self.assertEqual(
            set(containment),
            {"version", "mode", "relative_path", "device", "inode"},
        )
        self.assertEqual(containment["version"], 1)
        self.assertEqual(containment["mode"], "cgroup-v2")
        self.assertFalse(os.path.isabs(containment["relative_path"]))
        self.assertNotIn("..", containment["relative_path"].split("/"))
        self.assertGreater(containment["device"], 0)
        self.assertGreater(containment["inode"], 0)
        self.assertEqual(containment["relative_path"], expected_relative)
        self.assertEqual(
            self.supervisor.reconcile_leaf_intent(
                "deployment-a",
                "start:operation-a",
            ),
            containment,
        )

        descriptor = self.supervisor.reopen_verified(containment)
        os.close(descriptor)

    @requires_native_dirfd
    def test_create_leaf_refuses_to_reuse_an_existing_operation(self):
        self.supervisor.create_leaf("deployment-a", "operation-a")

        with self.assertRaisesRegex(RamdiskError, "already exists"):
            self.supervisor.create_leaf("deployment-a", "operation-a")

    @requires_native_dirfd
    def test_reopen_rejects_replaced_leaf(self):
        containment = self.supervisor.create_leaf("deployment-a", "operation-a")
        leaf = self.fixture.root / containment["relative_path"]
        old_descriptor = os.open(leaf, os.O_RDONLY | os.O_DIRECTORY)
        try:
            leaf.rmdir()
            leaf.mkdir()
        finally:
            os.close(old_descriptor)

        with self.assertRaisesRegex(RamdiskError, "identity.*changed"):
            self.supervisor.reopen_verified(containment)
        with self.assertRaisesRegex(RamdiskError, "identity.*changed"):
            self.supervisor.prove_removed(containment)
        leaf.rmdir()
        self.assertTrue(self.supervisor.prove_removed(containment))

        containment = self.supervisor.create_leaf(
            "deployment-a",
            "operation-after-proof",
        )
        leaf = self.fixture.materialize_leaf(
            containment,
            pids=(),
            populated=False,
        )
        original = self.fixture.root / "reviewed-leaf"
        replacement = self.fixture.root / "replacement-leaf"
        replacement.mkdir()
        prove_absence = self.supervisor.prove_absence

        def replace_after_proof(record):
            self.assertTrue(prove_absence(record))
            leaf.rename(original)
            replacement.rename(leaf)
            return True

        with mock.patch.object(
            self.supervisor,
            "prove_absence",
            side_effect=replace_after_proof,
        ):
            with self.assertRaisesRegex(RamdiskError, "identity.*changed"):
                self.supervisor.remove_empty(containment)

        self.assertTrue(leaf.is_dir())
        self.assertNotEqual(leaf.stat().st_ino, containment["inode"])

    @requires_native_dirfd
    def test_reopen_rejects_symlink_component(self):
        containment = self.supervisor.create_leaf("deployment-a", "operation-a")
        leaf = self.fixture.root / containment["relative_path"]
        target = self.fixture.root / "replacement"
        target.mkdir()
        leaf.rmdir()
        leaf.symlink_to(target, target_is_directory=True)

        with self.assertRaisesRegex(RamdiskError, "cannot reopen|symbolic"):
            self.supervisor.reopen_verified(containment)

    @requires_native_dirfd
    def test_delegated_root_rejects_symlinked_ancestor(self):
        actual = self.fixture.root / "actual-root"
        child = actual / "child"
        child.mkdir(parents=True)
        (child / "cgroup.controllers").write_text("memory\n")
        alias = self.fixture.root / "alias"
        alias.symlink_to(actual, target_is_directory=True)
        supervisor = supervision.CgroupSupervisor(root=str(alias / "child"))

        with self.assertRaisesRegex(RamdiskError, "symbolic|without symbolic"):
            supervisor.create_leaf("deployment-a", "operation-a")

    def test_default_discovery_fails_closed_without_unified_membership(self):
        with mock.patch(
            "builtins.open",
            return_value=io.StringIO("1:name=systemd:/user.slice\n"),
        ):
            with self.assertRaisesRegex(RamdiskError, "cannot discover"):
                supervision.CgroupSupervisor.discover_root()

    @requires_native_dirfd
    def test_missing_membership_is_inconclusive_not_absent(self):
        containment = self.supervisor.create_leaf("deployment-a", "operation-a")

        with self.assertRaises(supervision.ContainmentInconclusive):
            self.supervisor.members(containment)
        with self.assertRaises(supervision.ContainmentInconclusive):
            self.supervisor.prove_absence(containment)

    @requires_native_dirfd
    def test_populated_zero_and_stably_empty_membership_proves_absence(self):
        containment = self.supervisor.create_leaf("deployment-a", "operation-a")
        self.fixture.materialize_leaf(containment, pids=(), populated=False)

        self.assertTrue(self.supervisor.prove_absence(containment))

    @requires_native_dirfd
    def test_empty_procs_with_populated_one_is_inconclusive(self):
        containment = self.supervisor.create_leaf("deployment-a", "operation-a")
        self.fixture.materialize_leaf(containment, pids=(), populated=True)

        with self.assertRaisesRegex(
            supervision.ContainmentInconclusive,
            "populated",
        ):
            self.supervisor.prove_absence(containment)

        self.assertIsNone(self.supervisor.liveness({"containment": containment}))

    @requires_native_dirfd
    def test_membership_is_compared_as_a_kernel_set_not_file_order(self):
        containment = self.supervisor.create_leaf("deployment-a", "operation-a")
        leaf = self.fixture.materialize_leaf(containment, pids=(4101, 4100, 4101))
        (leaf / "cgroup.procs").write_text("4101\n4100\n4101\n")

        self.assertEqual(self.supervisor.members(containment), [4100, 4101])


class GatedExecTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def _command(self, marker):
        return [
            sys.executable,
            "-c",
            "from pathlib import Path; Path(%r).write_text('executed')"
            % str(marker),
        ]

    @requires_posix_pass_fds
    def test_gate_cannot_exec_before_release_and_keeps_the_same_pid(self):
        marker = self.root / "executed"
        gate = supervision.spawn_exec_gate(
            self._command(marker),
            environment={},
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self.addCleanup(supervision.abort_exec_gate, gate)

        time.sleep(0.05)
        self.assertIsNone(gate.process.poll())
        self.assertFalse(marker.exists())
        pid = gate.process.pid

        supervision.release_exec_gate(gate)
        self.assertEqual(gate.process.pid, pid)
        self.assertEqual(gate.process.wait(timeout=5), 0)
        self.assertEqual(marker.read_text(), "executed")

    @requires_posix_pass_fds
    def test_parent_eof_aborts_without_exec(self):
        marker = self.root / "executed"
        gate = supervision.spawn_exec_gate(
            self._command(marker),
            environment={},
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        supervision.abort_exec_gate(gate)
        self.assertNotEqual(gate.process.wait(timeout=5), 0)
        self.assertFalse(marker.exists())


class AttachAndSignalTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.fixture = CgroupFixture(self.temporary.name)

    @requires_native_dirfd
    def test_attach_opens_pidfd_and_revalidates_before_separate_release(self):
        opened = []
        supervisor = self.fixture.supervisor(
            pidfd_open=lambda pid, flags: opened.append((pid, flags)) or 91,
            pidfd_exited=lambda _fd: False,
            close_fd=lambda _fd: None,
        )
        containment = supervisor.create_leaf("deployment-a", "operation-a")
        leaf = self.fixture.materialize_leaf(containment)
        gate = mock.Mock()
        gate.process.pid = 4100
        gate.released = False
        gate.release_fd = 17
        gate.pidfd = None

        def write_member(_containment, pid):
            (leaf / "cgroup.procs").write_text("%d\n" % pid)
            (leaf / "cgroup.events").write_text("populated 1\n")

        with mock.patch.object(supervisor, "_write_member", write_member), mock.patch(
            "ramdisk_support.supervision.release_exec_gate"
        ) as release:
            supervisor.attach_gate(gate, containment)
            release.assert_not_called()
            supervisor.release_gate(gate, containment)

        self.assertEqual(opened, [(4100, 0)])
        release.assert_called_once_with(gate)

    @requires_native_dirfd
    def test_attach_refuses_pidfd_that_is_already_exited(self):
        supervisor = self.fixture.supervisor(
            pidfd_open=lambda _pid, _flags: 91,
            pidfd_exited=lambda _fd: True,
            close_fd=lambda _fd: None,
        )
        containment = supervisor.create_leaf("deployment-a", "operation-a")
        leaf = self.fixture.materialize_leaf(containment)
        gate = mock.Mock()
        gate.process.pid = 4100
        gate.pidfd = None

        def write_member(_containment, pid):
            (leaf / "cgroup.procs").write_text("%d\n" % pid)

        with mock.patch.object(supervisor, "_write_member", write_member):
            with self.assertRaisesRegex(RamdiskError, "pidfd.*exited"):
                supervisor.attach_gate(gate, containment)

    @requires_native_dirfd
    def test_attach_refuses_an_incomplete_cgroup_membership_write(self):
        supervisor = self.fixture.supervisor()
        containment = supervisor.create_leaf("deployment-a", "operation-a")
        self.fixture.materialize_leaf(containment)

        with mock.patch("ramdisk_support.supervision.os.write", return_value=0):
            with self.assertRaisesRegex(RamdiskError, "short write"):
                supervisor._write_member(containment, 4100)

    def test_close_gate_closes_both_pipe_and_pidfd(self):
        closed = []
        supervisor = self.fixture.supervisor(
            pidfd_open=lambda _pid, _flags: 91,
            close_fd=lambda fd: closed.append(fd),
        )
        gate = mock.Mock()
        gate.release_fd = None
        gate.pidfd = 91

        supervisor.close_gate(gate)

        self.assertEqual(closed, [91])
        self.assertIsNone(gate.pidfd)

    @requires_native_dirfd
    def test_verify_gate_rechecks_retained_pidfd_and_membership(self):
        supervisor = self.fixture.supervisor(
            pidfd_open=lambda _pid, _flags: 91,
            pidfd_exited=lambda _fd: False,
            close_fd=lambda _fd: None,
        )
        containment = supervisor.create_leaf("deployment-a", "operation-a")
        self.fixture.materialize_leaf(containment, pids=(4100, 4101))
        gate = mock.Mock()
        gate.process.pid = 4100
        gate.pidfd = 91

        self.assertTrue(supervisor.verify_gate(gate, containment))

    @requires_native_dirfd
    def test_record_verification_rechecks_membership_after_pidfd_bind(self):
        supervisor = self.fixture.supervisor(
            pidfd_open=lambda _pid, _flags: 91,
            pidfd_exited=lambda _fd: False,
            close_fd=lambda _fd: None,
        )
        containment = supervisor.create_leaf("deployment-a", "operation-a")
        record = {"pid": 4100, "containment": containment}

        with mock.patch.object(
            supervisor,
            "members",
            side_effect=([4100], []),
        ):
            verified, reason = supervisor.verify_record(record)

        self.assertFalse(verified)
        self.assertIn("changed containment", reason)

    @requires_native_dirfd
    def test_signal_pass_opens_and_validates_all_pidfds_before_first_signal(self):
        events = []
        supervisor = self.fixture.supervisor(
            pidfd_open=lambda pid, _flags: events.append(("open", pid)) or pid + 100,
            pidfd_send_signal=lambda fd, sig: events.append(("signal", fd, sig)),
            pidfd_exited=lambda fd: events.append(("poll", fd)) or False,
            close_fd=lambda fd: events.append(("close", fd)),
        )
        containment = supervisor.create_leaf("deployment-a", "operation-a")
        self.fixture.materialize_leaf(containment, pids=(4100, 4101))

        self.assertEqual(supervisor.signal_pass(containment, signal.SIGTERM), 2)
        first_signal = next(i for i, item in enumerate(events) if item[0] == "signal")
        self.assertEqual(
            [item for item in events[:first_signal] if item[0] == "open"],
            [("open", 4100), ("open", 4101)],
        )
        self.assertEqual(
            [item for item in events[:first_signal] if item[0] == "poll"],
            [
                ("poll", 4200),
                ("poll", 4201),
                ("poll", 4200),
                ("poll", 4201),
            ],
        )

    def test_terminate_reenumerates_and_catches_child_forked_during_term(self):
        sends = []
        memberships = iter(
            [
                [4100], [4100], [4100],
                [4100, 4101], [4100, 4101], [4100, 4101],
                [], [],
            ]
        )
        supervisor = self.fixture.supervisor(
            pidfd_open=lambda pid, _flags: pid + 100,
            pidfd_send_signal=lambda fd, sig: sends.append((fd - 100, sig)),
            pidfd_exited=lambda _fd: False,
            close_fd=lambda _fd: None,
            sleep=lambda _seconds: None,
        )
        containment = {
            "version": 1,
            "mode": "cgroup-v2",
            "relative_path": "colibri/fake/leaf",
            "device": 1,
            "inode": 2,
        }
        with mock.patch.object(
            supervisor,
            "members",
            side_effect=lambda _record: next(memberships),
        ), mock.patch.object(
            supervisor,
            "prove_absence",
            side_effect=(supervision.ContainmentInconclusive("live"),) * 2
            + (True,),
        ):
            supervisor.terminate(containment, term_seconds=1, kill_seconds=1)

        self.assertIn((4100, signal.SIGTERM), sends)
        self.assertIn((4101, signal.SIGTERM), sends)


class ContainmentRecordTest(unittest.TestCase):
    def test_validate_rejects_partial_and_boolean_identity(self):
        valid = {
            "version": 1,
            "mode": "cgroup-v2",
            "relative_path": "colibri/deployment/operation",
            "device": 10,
            "inode": 20,
        }
        self.assertEqual(supervision.validate_containment(valid), valid)
        for key in tuple(valid):
            broken = dict(valid)
            broken.pop(key)
            with self.subTest(key=key), self.assertRaises(RamdiskError):
                supervision.validate_containment(broken)
        for key in ("device", "inode"):
            broken = dict(valid)
            broken[key] = True
            with self.subTest(key=key), self.assertRaises(RamdiskError):
                supervision.validate_containment(broken)
        broken = dict(valid, version=True)
        with self.assertRaises(RamdiskError):
            supervision.validate_containment(broken)

    def test_json_round_trip_preserves_exact_identity(self):
        record = {
            "version": 1,
            "mode": "cgroup-v2",
            "relative_path": "colibri/deployment/operation",
            "device": 10,
            "inode": 20,
        }
        self.assertEqual(
            supervision.validate_containment(json.loads(json.dumps(record))),
            record,
        )


@unittest.skipUnless(
    os.environ.get("COLI_CGROUP_TEST_ROOT"),
    "requires an explicitly delegated COLI_CGROUP_TEST_ROOT",
)
class DelegatedCgroupLiveTest(unittest.TestCase):
    def test_create_prove_empty_and_remove_only_below_delegated_test_root(self):
        root = os.path.realpath(os.environ["COLI_CGROUP_TEST_ROOT"])
        self.assertNotIn(root, ("/", "/sys/fs/cgroup"))
        self.assertTrue(root.startswith("/sys/fs/cgroup/"))
        supervisor = supervision.CgroupSupervisor(root=root)
        operation = "live-%s-%s" % (os.getpid(), time.monotonic_ns())

        authority = supervisor.create_leaf("test-suite", operation)
        self.assertTrue(supervisor.prove_absence(authority))
        self.assertTrue(supervisor.remove_empty(authority))


if __name__ == "__main__":
    unittest.main()
