"""Durable dual-root workspace recovery for causal RAMMAP experiments."""

import copy
import contextlib
import hashlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


C_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(C_DIR))

from ramdisk_support import benchmark  # noqa: E402
from ramdisk_support import state as state_support  # noqa: E402
from ramdisk_support.common import RamdiskError  # noqa: E402

if __package__:
    from .ramdisk_test_support import canonical_temporary_directory  # noqa: E402
else:
    from ramdisk_test_support import canonical_temporary_directory  # noqa: E402


class WorkspaceFixture:
    def __init__(self, root):
        self.root = Path(root)
        self.model = self.root / "model"
        self.prepared = self.root / "prepared"
        self.state = self.root / "state"
        self.model.mkdir()
        self.prepared.mkdir()
        self.state.mkdir()
        self.shard = "model-00001.safetensors"
        (self.model / self.shard).write_bytes(b"canonical-shard")
        (self.prepared / self.shard).write_bytes(b"staged")
        staged_bytes = len(b"canonical-shard")
        size_bytes = max(
            staged_bytes + max(64 * benchmark.MIB, staged_bytes // 100),
            64 * benchmark.MIB,
        )
        prepared_identity = {
            "path": str(self.prepared),
            "filesystem": "tmpfs",
            "source": "tmpfs",
            "mount_id": 9,
            "parent_id": 1,
            "device": "9:1",
            "root": "/",
            "options": [],
            "super_options": [],
            "optional": [],
        }
        self.manifest = {
            "version": 1,
            "deployment_id": "d" * 32,
            "state": "stopped",
            "model_fingerprint": "sha256:" + "a" * 64,
            "plan": {
                "model": {
                    "path": str(self.model),
                    "fingerprint": "sha256:" + "a" * 64,
                },
                "source_shards": [
                    {
                        "name": self.shard,
                        "size_bytes": len(b"canonical-shard"),
                        "header_sha256": "b" * 64,
                    }
                ],
                "staging": {
                    "selected_shards": [self.shard],
                    "linked_shards": [],
                    "staged_bytes": staged_bytes,
                },
                "placement": {"memory_nodes": [0, 1]},
                "hardware": {
                    "online_nodes": [0, 1],
                    "nodes": [
                        {"id": 0, "reserve_bytes": 0},
                        {"id": 1, "reserve_bytes": 0},
                    ],
                },
                "mount_options": {
                    "thp": "advise",
                    "noswap": True,
                    "allow_swappable": False,
                },
                "reserve": {
                    "runtime_bytes": 0,
                    "page_table_bytes": 0,
                    "os_margin_bytes": 0,
                },
                "parallel": 1,
                "topology": "interleaved",
                "mounts": [
                    {
                        "path": str(self.prepared),
                        "node": None,
                        "policy": "interleave=static:0-1",
                        "size_bytes": size_bytes,
                    }
                ],
            },
            "mounts": [
                {
                    "path": str(self.prepared),
                    "node": None,
                    "policy": "interleave=static:0-1",
                    "size_bytes": size_bytes,
                    "ownership": "managed",
                    "operation_id": "d" * 32 + ":mount:0",
                    "identity": copy.deepcopy(prepared_identity),
                    "numa_allocation": {"0": 64, "1": 64},
                    "requested": {
                        "filesystem": "tmpfs",
                        "source": "tmpfs",
                        "size_bytes": size_bytes,
                        "policy": "interleave=static:0-1",
                        "thp": "advise",
                        "noswap": True,
                        "safety_options": [
                            "noatime", "nodev", "nosuid", "noexec", "mode=0700"
                        ],
                    },
                }
            ],
            "processes": [],
            "pending_launches": [],
        }
        self.mounts = {str(self.prepared): prepared_identity}
        self.snapshots = []
        self.events = []
        self.next_mount_id = 10
        self.busy = set()
        self.fail_unmount = set()
        self.populate_sources = []
        self.forced_scratch_device = None

    def save(self, manifest):
        self.snapshots.append(copy.deepcopy(manifest))

    def mount_tmpfs(self, plan, record):
        workspace = self.manifest["benchmark_workspace"]
        durable = next(
            item for item in workspace["roots"] if item["path"] == record["path"]
        )
        self.events.append(("mount", durable["ownership"], workspace["phase"]))
        identity = {
            "path": record["path"],
            "filesystem": "tmpfs",
            "source": "tmpfs",
            "mount_id": self.next_mount_id,
            "parent_id": 1,
            "device": self.forced_scratch_device or "%d:1" % self.next_mount_id,
            "root": "/",
            "options": [
                "noatime",
                "nodev",
                "nosuid",
                "noexec",
                "mode=700",
                "huge=advise",
                "noswap",
                "mpol=" + record["policy"],
            ],
            "super_options": [],
            "optional": [],
        }
        self.next_mount_id += 1
        self.mounts[record["path"]] = identity
        record["effective_thp"] = "advise"
        record["effective_noswap"] = True

    def validate_mount(self, record, _plan):
        return copy.deepcopy(self.mounts[record["path"]])

    def populate_mount(self, _plan, record, **kwargs):
        durable = next(
            item
            for item in self.manifest["benchmark_workspace"]["roots"]
            if item["path"] == record["path"]
        )
        self.events.append(("stage", durable["stage_phase"]))
        self.populate_sources.append(kwargs.get("source_root"))
        (Path(record["path"]) / self.shard).write_bytes(b"staged")

    def validate_namespace(self, _plan, _record, sample_numa=True):
        self.events.append(("namespace", sample_numa))
        return {"0": 64}

    def umount(self, path, _hardware):
        workspace = self.manifest["benchmark_workspace"]
        self.events.append(
            (
                "umount",
                workspace["phase"],
                all(
                    root.get("cleanup_authorized_at")
                    for root in workspace["roots"]
                    if root.get("role") == "scratch"
                ),
            )
        )
        if path in self.fail_unmount:
            raise RamdiskError("forced scratch unmount failure")
        self.mounts.pop(path, None)
        # The fake mount writes into the visible mountpoint. A real unmount
        # reveals the empty underlying directory, so mirror that transition.
        for child in Path(path).iterdir():
            child.unlink()

    def manager(
        self,
        *,
        available_bytes=1 << 40,
        process_supervisor=None,
        process_supervisor_factory=None,
        process_starttime=None,
    ):
        return benchmark.DurableWorkspaceManager(
            load_manifest=lambda required=True: self.manifest,
            save_manifest=self.save,
            state_root=lambda: str(self.state),
            ensure_private_dir=lambda path: Path(path).mkdir(
                mode=0o700,
                parents=True,
                exist_ok=True,
            ),
            assert_durable_state_dir=lambda _path, plan=None: None,
            mount_at=lambda path: copy.deepcopy(self.mounts.get(path)),
            mount_table=lambda: [copy.deepcopy(item) for item in self.mounts.values()],
            path_is_below=lambda child, parent: os.path.commonpath(
                [os.path.abspath(child), os.path.abspath(parent)]
            ) == os.path.abspath(parent) and os.path.abspath(child) != os.path.abspath(parent),
            busy_mount_references=lambda path, hardware=None: (
                [999] if path in self.busy else []
            ),
            mount_tmpfs=self.mount_tmpfs,
            validate_mount=self.validate_mount,
            populate_mount=self.populate_mount,
            validate_namespace=self.validate_namespace,
            source_still_matches=lambda _plan: None,
            umount_path=self.umount,
            available_for_mount=lambda _mount, plan=None: available_bytes,
            process_supervisor=process_supervisor,
            process_supervisor_factory=process_supervisor_factory,
            process_starttime=process_starttime,
            uid_provider=lambda: 1000,
        )


class FakeGateProcess:
    def __init__(self, pid):
        self.pid = pid
        self.returncode = None
        self.stdin = None
        self.stdout = None

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        del timeout
        return self.returncode


class FakeGate:
    def __init__(self, pid):
        self.process = FakeGateProcess(pid)
        self.released = False


class FakeProcessSupervisor:
    def __init__(self, fixture):
        self.fixture = fixture
        self.events = []
        self.environment = None
        self.command = None
        self.stdout_payload = None
        self.containment = {
            "version": 1,
            "mode": "cgroup-v2",
            "relative_path": "colibri/dtest/oreplicate",
            "device": 41,
            "inode": 42,
        }
        self.live = False
        self.leaf_exists = False
        self.removed = False

    def create_leaf(self, deployment_id, operation_id):
        self.containment["relative_path"] = "colibri/d%s/o%s" % (
            hashlib.sha256(deployment_id.encode("utf-8")).hexdigest()[:24],
            hashlib.sha256(operation_id.encode("utf-8")).hexdigest()[:24],
        )
        self.events.append(("create", deployment_id, operation_id))
        self.leaf_exists = True
        return copy.deepcopy(self.containment)

    def reconcile_leaf_intent(self, _deployment_id, _operation_id):
        return (
            copy.deepcopy(self.containment)
            if self.leaf_exists and not self.removed
            else None
        )

    def spawn_gate(self, command, *, environment, **_kwargs):
        self.command = list(command)
        self.environment = dict(environment)
        self.events.append(("spawn",))
        self.live = True
        gate = FakeGate(4321)
        try:
            import openai_server

            gate.process.stdin = io.BytesIO()
            payload = self.stdout_payload
            if payload is None:
                payload = openai_server.READY + b"STAT 0 0 0 0\n"
            gate.process.stdout = io.BytesIO(payload)
        except ImportError:
            pass
        self.gate = gate
        return gate

    def attach_gate(self, gate, containment):
        self.events.append(("attach", gate.process.pid, containment["inode"]))

    def release_gate(self, gate, _containment):
        pending = self.fixture.manifest["benchmark_workspace"]["pending_process"]
        self.events.append(("release", pending["containment_phase"]))
        gate.released = True

    def verify_gate(self, _gate, _containment):
        return True

    def close_gate(self, _gate):
        self.events.append(("close-gate",))

    def abort_gate(self, _gate):
        self.events.append(("abort-gate",))
        self.live = False
        _gate.process.returncode = 1

    def verify_record(self, record):
        return (
            self.live
            and record.get("pid") == 4321
            and record.get("containment") == self.containment,
            None,
        )

    def terminate(self, _containment):
        self.events.append(("terminate",))
        self.fixture.events.append(("process-terminate",))
        self.live = False
        if hasattr(self, "gate"):
            self.gate.process.returncode = 0
        return {"status": "absent"}

    def prove_absence(self, _containment):
        if self.live:
            raise RamdiskError("fake cgroup remains populated")
        return True

    def prove_removed(self, containment):
        if self.leaf_exists:
            if getattr(self, "replacement", False):
                raise RamdiskError("cgroup containment identity changed")
            if containment != self.containment:
                raise RamdiskError("cgroup containment identity changed")
            raise RamdiskError(
                "cgroup containment remains present after removal"
            )
        return True

    def remove_empty(self, _containment):
        self.events.append(("remove",))
        self.fixture.events.append(("process-remove",))
        self.removed = True
        self.leaf_exists = False
        return True


class DurableWorkspaceTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.temporary_root = str(Path(self.temporary.name).resolve())
        self.fixture = WorkspaceFixture(self.temporary_root)

    def test_open_journals_every_irreversible_transition_and_cleans(self):
        manager = self.fixture.manager()

        with manager.open(self.fixture.manifest, {"protocol_id": "a" * 64}, None) as roots:
            self.assertEqual(set(roots), {"interleaved", "local"})
            self.assertNotEqual(roots["interleaved"]["identity"], roots["local"]["identity"])
            self.assertEqual(
                self.fixture.manifest["benchmark_workspace"]["phase"],
                "staged",
            )
            self.assertTrue(manager.verify(roots["interleaved"]))
            self.assertTrue(manager.verify(roots["local"]))

        self.assertNotIn("benchmark_workspace", self.fixture.manifest)
        self.assertEqual(
            [item for item in self.fixture.events if item[0] == "mount"],
            [("mount", "pending", "pending")],
        )
        self.assertEqual(self.fixture.populate_sources, [str(self.fixture.prepared)])

    def test_workspace_without_pending_process_does_not_discover_cgroups(self):
        discoveries = []
        manager = self.fixture.manager(
            process_supervisor_factory=lambda: discoveries.append(True)
        )

        with manager.open(
            self.fixture.manifest,
            {"protocol_id": "a" * 64},
            None,
        ):
            pass

        self.assertEqual(discoveries, [])
        self.assertEqual(
            [item for item in self.fixture.events if item[0] == "stage"],
            [("stage", "pending")],
        )
        self.assertIn(str(self.fixture.prepared), self.fixture.mounts)
        self.assertTrue(
            all(item == ("umount", "cleanup", True) for item in self.fixture.events if item[0] == "umount")
        )

    def test_cleanup_preflights_all_roots_before_first_unmount(self):
        manager = self.fixture.manager()
        context = manager.open(self.fixture.manifest, {"protocol_id": "a" * 64}, None)
        roots = context.__enter__()
        self.fixture.busy.add(roots["local"]["path"])

        with self.assertRaisesRegex(benchmark.WorkspaceCleanupError, "busy"):
            context.__exit__(None, None, None)

        self.assertFalse(any(item[0] == "umount" for item in self.fixture.events))
        self.assertEqual(self.fixture.manifest["benchmark_workspace"]["phase"], "cleanup")

    def test_capacity_is_admitted_before_workspace_journal_or_mount(self):
        manager = self.fixture.manager(available_bytes=0)
        with self.assertRaisesRegex(RamdiskError, "additional bytes"):
            with manager.open(
                self.fixture.manifest, {"protocol_id": "a" * 64}, None
            ):
                self.fail("capacity rejection must happen before open yields")
        self.assertNotIn("benchmark_workspace", self.fixture.manifest)
        self.assertFalse(any(event[0] == "mount" for event in self.fixture.events))

    def test_scratch_must_use_a_distinct_filesystem_device(self):
        self.fixture.forced_scratch_device = "9:1"
        with self.assertRaisesRegex(RamdiskError, "distinct devices"):
            with self.fixture.manager().open(
                self.fixture.manifest, {"protocol_id": "a" * 64}, None
            ):
                self.fail("same-device scratch must not be exposed")
        self.assertIn(str(self.fixture.prepared), self.fixture.mounts)

    def test_per_node_plan_borrows_node_zero_and_builds_only_interleaved_scratch(self):
        plan = self.fixture.manifest["plan"]
        plan["topology"] = "per-node"
        size = plan["mounts"][0]["size_bytes"]
        plan["mounts"][0].update(
            node=0,
            policy="bind=static:0",
        )
        prepared = self.fixture.manifest["mounts"][0]
        prepared.update(node=0, policy="bind=static:0")
        prepared["requested"]["policy"] = "bind=static:0"
        with self.fixture.manager().open(
            self.fixture.manifest, {"protocol_id": "a" * 64}, None
        ) as roots:
            self.assertEqual(roots["local"]["role"], "deployment")
            self.assertEqual(roots["local"]["path"], str(self.fixture.prepared))
            self.assertEqual(roots["interleaved"]["role"], "scratch")
            self.assertEqual(roots["interleaved"]["size_bytes"], size)
        self.assertEqual(self.fixture.populate_sources, [str(self.fixture.prepared)])

    def test_unmount_failure_keeps_durable_cleanup_authority(self):
        manager = self.fixture.manager()
        context = manager.open(self.fixture.manifest, {"protocol_id": "a" * 64}, None)
        roots = context.__enter__()
        self.fixture.fail_unmount.add(roots["local"]["path"])

        with self.assertRaisesRegex(benchmark.WorkspaceCleanupError, "forced"):
            context.__exit__(None, None, None)

        workspace = self.fixture.manifest["benchmark_workspace"]
        self.assertEqual(workspace["phase"], "cleanup")
        scratch = next(root for root in workspace["roots"] if root["role"] == "scratch")
        deployment = next(
            root for root in workspace["roots"] if root["role"] == "deployment"
        )
        self.assertTrue(scratch.get("cleanup_authorized_at"))
        self.assertNotIn("cleanup_authorized_at", deployment)
        self.assertIn(deployment["path"], self.fixture.mounts)

    def test_pending_helper_outcome_is_never_treated_as_absence(self):
        manager = self.fixture.manager()
        context = manager.open(
            self.fixture.manifest, {"protocol_id": "a" * 64}, None
        )
        context.__enter__()
        snapshot = next(
            copy.deepcopy(item)
            for item in self.fixture.snapshots
            if (item.get("benchmark_workspace") or {}).get("phase") == "pending"
            and any(
                root.get("role") == "scratch"
                and root.get("helper_started_at")
                and not root.get("helper_completed_at")
                for root in item["benchmark_workspace"]["roots"]
            )
        )
        context.__exit__(None, None, None)
        scratch = next(
            root for root in snapshot["benchmark_workspace"]["roots"]
            if root["role"] == "scratch"
        )
        Path(scratch["path"]).mkdir(parents=True, exist_ok=True)
        self.fixture.mounts.pop(scratch["path"], None)
        self.fixture.manifest = snapshot

        with self.assertRaisesRegex(benchmark.WorkspaceCleanupError, "outcome is unknown"):
            manager.recover(self.fixture.manifest)

        self.assertIn("benchmark_workspace", self.fixture.manifest)

    def test_authorized_cleanup_recovers_crash_after_unmount_before_save(self):
        manager = self.fixture.manager()
        context = manager.open(
            self.fixture.manifest, {"protocol_id": "a" * 64}, None
        )
        context.__enter__()
        workspace = self.fixture.manifest["benchmark_workspace"]
        workspace["phase"] = "cleanup"
        scratch = next(root for root in workspace["roots"] if root["role"] == "scratch")
        scratch["cleanup_authorized_at"] = "9999-12-31T23:59:59+00:00"
        self.fixture.mounts.pop(scratch["path"], None)
        for child in Path(scratch["path"]).iterdir():
            child.unlink()
        self.fixture.save(self.fixture.manifest)

        self.assertTrue(manager.recover(self.fixture.manifest))
        self.assertNotIn("benchmark_workspace", self.fixture.manifest)

    def test_reload_recovers_after_paths_removed_before_final_manifest_save(self):
        manager = self.fixture.manager()
        context = manager.open(
            self.fixture.manifest,
            {"protocol_id": "a" * 64},
            None,
        )
        context.__enter__()
        workspace = self.fixture.manifest["benchmark_workspace"]
        workspace["phase"] = "cleanup"
        stamp = "9999-12-31T23:59:59+00:00"
        scratch = next(root for root in workspace["roots"] if root["role"] == "scratch")
        scratch["cleanup_authorized_at"] = stamp
        self.fixture.mounts.pop(scratch["path"], None)
        for child in Path(scratch["path"]).iterdir():
            child.unlink()
        Path(scratch["path"]).rmdir()
        scratch["unmounted_at"] = stamp
        scratch["removed_at"] = stamp
        Path(workspace["operation_path"]).rmdir()
        self.fixture.save(self.fixture.manifest)

        reloaded = copy.deepcopy(self.fixture.snapshots[-1])
        state_support._validate_benchmark_workspace(
            reloaded["benchmark_workspace"],
            manifest=reloaded,
            state_root=lambda: str(self.fixture.state),
        )
        self.fixture.manifest = reloaded
        self.assertTrue(manager.recover(reloaded))
        self.assertNotIn("benchmark_workspace", reloaded)

    def test_verify_rejects_replaced_mount_identity(self):
        manager = self.fixture.manager()
        with manager.open(self.fixture.manifest, {"protocol_id": "a" * 64}, None) as roots:
            descriptor = roots["interleaved"]
            self.fixture.mounts[descriptor["path"]]["mount_id"] += 100
            self.assertFalse(manager.verify(descriptor))
            self.fixture.mounts[descriptor["path"]]["mount_id"] -= 100

    def test_state_validation_binds_exact_workspace_policy(self):
        manager = self.fixture.manager()
        context = manager.open(
            self.fixture.manifest,
            {"protocol_id": "a" * 64},
            None,
        )
        context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)
        workspace = copy.deepcopy(
            self.fixture.manifest["benchmark_workspace"]
        )

        state_support._validate_benchmark_workspace(
            workspace,
            manifest=self.fixture.manifest,
            state_root=lambda: str(self.fixture.state),
        )
        workspace["roots"][0]["policy"] = "bind=static:1"
        with self.assertRaisesRegex(RamdiskError, "workspace root"):
            state_support._validate_benchmark_workspace(
                workspace,
                manifest=self.fixture.manifest,
                state_root=lambda: str(self.fixture.state),
            )

        numeric_protocol = copy.deepcopy(
            self.fixture.manifest["benchmark_workspace"]
        )
        numeric_protocol["protocol_id"] = 123
        with self.assertRaisesRegex(RamdiskError, "invalid benchmark workspace"):
            state_support._validate_benchmark_workspace(
                numeric_protocol,
                manifest=self.fixture.manifest,
                state_root=lambda: str(self.fixture.state),
            )

        same_device = copy.deepcopy(
            self.fixture.manifest["benchmark_workspace"]
        )
        same_device["roots"][1]["identity"]["device"] = (
            same_device["roots"][0]["identity"]["device"]
        )
        with self.assertRaisesRegex(RamdiskError, "physically distinct"):
            state_support._validate_benchmark_workspace(
                same_device,
                manifest=self.fixture.manifest,
                state_root=lambda: str(self.fixture.state),
            )

        changed_plan = copy.deepcopy(self.fixture.manifest)
        changed_plan["plan"]["mount_options"]["noswap"] = False
        with self.assertRaisesRegex(RamdiskError, "mount policy"):
            state_support._validate_benchmark_workspace(
                copy.deepcopy(self.fixture.manifest["benchmark_workspace"]),
                manifest=changed_plan,
                state_root=lambda: str(self.fixture.state),
            )

        for name, value in (("noswap", "false"), ("thp", 1)):
            malformed_policy = copy.deepcopy(self.fixture.manifest)
            malformed_policy["plan"]["mount_options"][name] = value
            with self.subTest(name=name), self.assertRaisesRegex(
                RamdiskError,
                "mount options",
            ):
                state_support._validate_benchmark_workspace(
                    copy.deepcopy(
                        self.fixture.manifest["benchmark_workspace"]
                    ),
                    manifest=malformed_policy,
                    state_root=lambda: str(self.fixture.state),
                )

        boolean_node = copy.deepcopy(
            self.fixture.manifest["benchmark_workspace"]
        )
        local = next(
            root for root in boolean_node["roots"]
            if root["name"] == "local"
        )
        local.update(nodes=[False], node=False)
        with self.assertRaisesRegex(RamdiskError, "workspace root"):
            state_support._validate_benchmark_workspace(
                boolean_node,
                manifest=self.fixture.manifest,
                state_root=lambda: str(self.fixture.state),
            )

        for name, value in (("effective_noswap", "false"), ("effective_thp", 1)):
            malformed_effective = copy.deepcopy(
                self.fixture.manifest["benchmark_workspace"]
            )
            scratch = next(
                root for root in malformed_effective["roots"]
                if root["role"] == "scratch"
            )
            scratch[name] = value
            scratch["requested"][name] = value
            with self.subTest(name=name), self.assertRaisesRegex(
                RamdiskError,
                "effective mount policy",
            ):
                state_support._validate_benchmark_workspace(
                    malformed_effective,
                    manifest=self.fixture.manifest,
                    state_root=lambda: str(self.fixture.state),
                )

        missing_deployment = copy.deepcopy(self.fixture.manifest)
        missing_deployment["deployment_id"] = None
        with self.assertRaisesRegex(RamdiskError, "invalid benchmark workspace"):
            state_support._validate_benchmark_workspace(
                copy.deepcopy(self.fixture.manifest["benchmark_workspace"]),
                manifest=missing_deployment,
                state_root=lambda: str(self.fixture.state),
            )


class BenchmarkProcessSupervisionTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.temporary_root = str(Path(self.temporary.name).resolve())
        self.fixture = WorkspaceFixture(self.temporary_root)
        self.supervisor = FakeProcessSupervisor(self.fixture)

    def _starttime(self, pid):
        if pid == 4321 and self.supervisor.live:
            return 9001
        raise FileNotFoundError("process is absent")

    def _metadata(self, roots):
        protocol_id = "a" * 64
        treatment_id = "tmpfs-rammap-local"
        launch_id = "b" * 32
        state_dir = (
            self.fixture.state
            / "causal-benchmark-state"
            / protocol_id
            / ("0000-003-%s-%s" % (treatment_id, launch_id))
        )
        state_dir.mkdir(parents=True, exist_ok=True)
        return {
            "protocol_id": protocol_id,
            "treatment_id": treatment_id,
            "block_index": 0,
            "sequence": 3,
            "launch_id": launch_id,
            "state_dir": str(state_dir),
            "weights_dir": roots["local"]["path"],
        }

    def _pending_record(self, roots, phase, *, launch_id="b" * 32):
        metadata = self._metadata(roots)
        metadata["launch_id"] = launch_id
        operation_id = "replicate:" + launch_id
        containment = None
        pid = None
        starttime = None
        if phase != "create-intent":
            containment = copy.deepcopy(self.supervisor.containment)
            containment["relative_path"] = "colibri/d%s/o%s" % (
                hashlib.sha256(
                    self.fixture.manifest["deployment_id"].encode("utf-8")
                ).hexdigest()[:24],
                hashlib.sha256(operation_id.encode("utf-8")).hexdigest()[:24],
            )
        if phase not in ("create-intent", "cgroup-created"):
            pid = 4321
            starttime = 9001
        return {
            "version": 1,
            "operation_id": operation_id,
            "workspace_operation_id": self.fixture.manifest[
                "benchmark_workspace"
            ]["operation_id"],
            "protocol_id": metadata["protocol_id"],
            "treatment_id": metadata["treatment_id"],
            "block_index": metadata["block_index"],
            "sequence": metadata["sequence"],
            "launch_id": launch_id,
            "uid": 1000,
            "state_dir": metadata["state_dir"],
            "weights_dir": metadata["weights_dir"],
            "expected_command": ["/opt/colibri", "8"],
            "environment_sha256": "e" * 64,
            "containment": containment,
            "containment_phase": phase,
            "pid": pid,
            "starttime": starttime,
            "created_at": "2026-08-08T00:00:00+00:00",
        }

    def test_gate_release_follows_durable_attach_and_evidence_binds_exact_pid(self):
        manager = self.fixture.manager(
            process_supervisor=self.supervisor,
            process_starttime=self._starttime,
        )
        with manager.open(
            self.fixture.manifest, {"protocol_id": "a" * 64}, None
        ) as roots:
            metadata = self._metadata(roots)
            with manager.process_attempt(
                recheck=lambda: self.supervisor.events.append(("recheck",)),
                **metadata
            ) as attempt:
                process = attempt.spawn(
                    ["/usr/bin/numactl", "/opt/colibri", "8"],
                    child_env={
                        "SNAP": roots["local"]["path"],
                        "COLI_STATE_DIR": metadata["state_dir"],
                        "SERVE": "1",
                    },
                    stderr=None,
                )
                child_environment = {
                    "SNAP": roots["local"]["path"],
                    "COLI_STATE_DIR": metadata["state_dir"],
                    "SERVE": "1",
                }
                attempt.retire(process)
                row = {
                    "process": {"pid": process.pid, "starttime": 9001},
                    "applied_environment": child_environment,
                }
                attempt.bind_evidence(row)
                self.assertEqual(
                    row["process"]["supervision_mode"], "cgroup-v2"
                )
                self.assertRegex(
                    row["process"]["containment_identity_sha256"],
                    r"^[0-9a-f]{64}$",
                )
                self.assertNotIn("nonce", row["process"])

            self.assertNotIn(
                "pending_process",
                self.fixture.manifest["benchmark_workspace"],
            )

        self.assertIn(("release", "attached-verified"), self.supervisor.events)
        self.assertEqual(
            [event[0] for event in self.supervisor.events].count("recheck"), 2
        )
        self.assertEqual(self.supervisor.environment, child_environment)
        self.assertNotIn("COLI_MANAGED_NONCE", self.supervisor.environment)
        phases = [
            snapshot["benchmark_workspace"]["pending_process"]["containment_phase"]
            for snapshot in self.fixture.snapshots
            if (snapshot.get("benchmark_workspace") or {}).get("pending_process")
            and snapshot["benchmark_workspace"]["pending_process"].get(
                "containment_phase"
            )
        ]
        self.assertEqual(
            phases[:5],
            [
                "create-intent",
                "cgroup-created",
                "gate-spawned",
                "attached-verified",
                "gate-released",
            ],
        )
        self.assertLess(
            self.fixture.events.index(("process-terminate",)),
            next(
                index for index, event in enumerate(self.fixture.events)
                if event[0] == "umount"
            ),
        )

    def test_engine_spawn_and_close_use_supervised_hooks_end_to_end(self):
        manager = self.fixture.manager(
            process_supervisor=self.supervisor,
            process_starttime=self._starttime,
        )
        with manager.open(
            self.fixture.manifest, {"protocol_id": "a" * 64}, None
        ) as roots:
            metadata = self._metadata(roots)
            with manager.process_attempt(
                recheck=lambda: None,
                **metadata
            ) as attempt:
                engine_type, _render, _cancelled = (
                    benchmark._default_engine_dependencies(
                        None,
                        spawn_process=attempt.spawn,
                        terminate_process=attempt.retire,
                    )
                )
                engine = engine_type(
                    "/opt/colibri",
                    roots["local"]["path"],
                    max_tokens=32,
                    env={"COLI_STATE_DIR": metadata["state_dir"]},
                )
                applied_environment = copy.deepcopy(
                    engine.benchmark_child_environment
                )
                engine.close()
                row = {
                    "process": {"pid": 4321, "starttime": 9001},
                    "applied_environment": applied_environment,
                }
                attempt.bind_evidence(row)

            self.assertEqual(
                self.supervisor.environment,
                applied_environment,
            )
            self.assertNotIn("COLI_MANAGED_NONCE", applied_environment)
            self.assertEqual(row["process"]["launch_id"], "b" * 32)
            self.assertNotIn(
                "pending_process",
                self.fixture.manifest["benchmark_workspace"],
            )

    def test_constructor_or_ready_failure_retires_containment_before_workspace(self):
        manager = self.fixture.manager(
            process_supervisor=self.supervisor,
            process_starttime=self._starttime,
        )
        self.supervisor.stdout_payload = b""
        with self.assertRaisesRegex(RuntimeError, "exited unexpectedly"):
            with manager.open(
                self.fixture.manifest, {"protocol_id": "a" * 64}, None
            ) as roots:
                metadata = self._metadata(roots)
                with manager.process_attempt(
                    recheck=lambda: None,
                    **metadata
                ) as attempt:
                    engine_type, _render, _cancelled = (
                        benchmark._default_engine_dependencies(
                            None,
                            spawn_process=attempt.spawn,
                            terminate_process=attempt.retire,
                        )
                    )
                    engine_type(
                        "/opt/colibri",
                        roots["local"]["path"],
                        max_tokens=32,
                        env={"COLI_STATE_DIR": metadata["state_dir"]},
                    )
        self.assertTrue(self.supervisor.removed)
        self.assertNotIn("benchmark_workspace", self.fixture.manifest)

    def test_create_intent_recovery_adopts_only_an_exact_empty_leaf(self):
        manager = self.fixture.manager(
            process_supervisor=self.supervisor,
            process_starttime=self._starttime,
        )
        roots = manager._prepare(
            self.fixture.manifest, {"protocol_id": "a" * 64}, None
        )
        record = self._pending_record(roots, "create-intent")
        self.fixture.manifest["benchmark_workspace"]["pending_process"] = record

        manager.recover(self.fixture.manifest)
        self.assertFalse(
            any(event[0] == "terminate" for event in self.supervisor.events)
        )
        self.assertNotIn("benchmark_workspace", self.fixture.manifest)

        # A fresh operation with the deterministic leaf already created may be
        # adopted only after proving that it has no members.
        adopt_root = Path(self.temporary_root) / "adopt"
        adopt_root.mkdir()
        self.fixture = WorkspaceFixture(str(adopt_root))
        self.supervisor = FakeProcessSupervisor(self.fixture)
        manager = self.fixture.manager(
            process_supervisor=self.supervisor,
            process_starttime=self._starttime,
        )
        roots = manager._prepare(
            self.fixture.manifest, {"protocol_id": "a" * 64}, None
        )
        record = self._pending_record(roots, "create-intent")
        self.fixture.manifest["benchmark_workspace"]["pending_process"] = record
        self.supervisor.containment = copy.deepcopy(
            self._pending_record(roots, "cgroup-created")["containment"]
        )
        self.supervisor.leaf_exists = True

        manager.recover(self.fixture.manifest)
        self.assertTrue(self.supervisor.removed)
        self.assertNotIn("benchmark_workspace", self.fixture.manifest)

    def test_populated_create_intent_is_retained_without_unmount_or_signal(self):
        manager = self.fixture.manager(
            process_supervisor=self.supervisor,
            process_starttime=self._starttime,
        )
        roots = manager._prepare(
            self.fixture.manifest, {"protocol_id": "a" * 64}, None
        )
        record = self._pending_record(roots, "create-intent")
        self.fixture.manifest["benchmark_workspace"]["pending_process"] = record
        self.supervisor.containment = copy.deepcopy(
            self._pending_record(roots, "cgroup-created")["containment"]
        )
        self.supervisor.leaf_exists = True
        self.supervisor.live = True

        with self.assertRaisesRegex(
            benchmark.WorkspaceCleanupError,
            "populated",
        ):
            manager.recover(self.fixture.manifest)
        self.assertIn(
            "pending_process",
            self.fixture.manifest["benchmark_workspace"],
        )
        self.assertFalse(
            any(event[0] == "terminate" for event in self.supervisor.events)
        )
        self.assertFalse(
            any(event[0] == "umount" for event in self.fixture.events)
        )

    def test_hard_crash_reloads_each_durable_gate_phase_with_no_live_handle(self):
        manager = self.fixture.manager(
            process_supervisor=self.supervisor,
            process_starttime=self._starttime,
        )
        roots = manager._prepare(
            self.fixture.manifest, {"protocol_id": "a" * 64}, None
        )
        metadata = self._metadata(roots)
        with manager.process_attempt(
            recheck=lambda: None,
            **metadata
        ) as attempt:
            process = attempt.spawn(
                ["/opt/colibri", "8"],
                child_env={
                    "SNAP": roots["local"]["path"],
                    "COLI_STATE_DIR": metadata["state_dir"],
                },
                stderr=None,
            )
            durable_by_phase = {
                snapshot["benchmark_workspace"]["pending_process"][
                    "containment_phase"
                ]: copy.deepcopy(snapshot)
                for snapshot in self.fixture.snapshots
                if (snapshot.get("benchmark_workspace") or {}).get(
                    "pending_process"
                )
            }
            attempt.retire(process)

        self.assertEqual(
            set(durable_by_phase),
            {
                "create-intent",
                "cgroup-created",
                "gate-spawned",
                "attached-verified",
                "gate-released",
            },
        )
        crash_cases = (
            # before-commit means the kernel side effect happened while the
            # preceding phase is the last durable copy.  Unreleased gates die
            # on parent EOF; released gates may still be executing.
            ("gate-spawn-before-commit", "cgroup-created", False),
            ("gate-spawn-after-commit", "gate-spawned", False),
            ("attach-before-commit", "gate-spawned", False),
            ("attach-after-commit", "attached-verified", False),
            ("release-before-commit", "attached-verified", True),
            ("release-after-commit", "gate-released", True),
        )
        for label, durable_phase, process_live in crash_cases:
            with self.subTest(label=label):
                # Discard every live manager/gate object and reload only the
                # last copy-on-success manifest snapshot.
                self.fixture.manifest = copy.deepcopy(
                    durable_by_phase[durable_phase]
                )
                fresh_supervisor = FakeProcessSupervisor(self.fixture)
                record = self.fixture.manifest["benchmark_workspace"][
                    "pending_process"
                ]
                fresh_supervisor.containment = copy.deepcopy(
                    record["containment"]
                )
                fresh_supervisor.leaf_exists = True
                fresh_supervisor.live = process_live

                def starttime(pid):
                    if pid == 4321 and fresh_supervisor.live:
                        return 9001
                    raise FileNotFoundError("process is absent")

                fresh_manager = self.fixture.manager(
                    process_supervisor=fresh_supervisor,
                    process_starttime=starttime,
                )
                fresh_manager._recover_pending_process(
                    self.fixture.manifest,
                    gate=None,
                )
                self.assertNotIn(
                    "pending_process",
                    self.fixture.manifest["benchmark_workspace"],
                )
                self.assertTrue(fresh_supervisor.removed)
                self.assertFalse(fresh_supervisor.live)
                self.assertFalse(
                    any(
                        event[0] == "release"
                        for event in fresh_supervisor.events
                    )
                )
                # A second recovery from the same durable result is a no-op.
                fresh_manager._recover_pending_process(
                    self.fixture.manifest,
                    gate=None,
                )

    @mock.patch.object(state_support, "current_uid", return_value=1000)
    def test_pending_process_state_rejects_spliced_or_mistyped_authority(
        self, _current_uid
    ):
        manager = self.fixture.manager(
            process_supervisor=self.supervisor,
            process_starttime=self._starttime,
        )
        roots = manager._prepare(
            self.fixture.manifest, {"protocol_id": "a" * 64}, None
        )
        workspace = self.fixture.manifest["benchmark_workspace"]
        workspace["pending_process"] = self._pending_record(
            roots, "gate-released"
        )
        state_support._validate_benchmark_workspace(
            workspace,
            manifest=self.fixture.manifest,
            state_root=lambda: str(self.fixture.state),
        )

        mutations = {
            "workspace operation": lambda record: record.update(
                workspace_operation_id="benchmark:" + "f" * 32
            ),
            "protocol": lambda record: record.update(protocol_id="f" * 64),
            "launch": lambda record: record.update(launch_id="c" * 32),
            "block sequence": lambda record: record.update(sequence=7),
            "state path": lambda record: record.update(state_dir="/tmp/escape"),
            "weights path": lambda record: record.update(
                weights_dir=str(self.fixture.model)
            ),
            "containment operation": lambda record: record[
                "containment"
            ].update(relative_path="colibri/dwrong/owrong"),
            "boolean inode": lambda record: record["containment"].update(
                inode=True
            ),
            "phase identity": lambda record: record.update(
                containment_phase="cgroup-created"
            ),
            "non-UTC creation": lambda record: record.update(
                created_at="2026-08-08T01:00:00+01:00"
            ),
            "environment hash": lambda record: record.update(
                environment_sha256=True
            ),
            "uid": lambda record: record.update(uid=1001),
            "create-intent removal": lambda record: record.update(
                containment=None,
                containment_phase="create-intent",
                pid=None,
                starttime=None,
                containment_removal_authorized_at=(
                    "2026-08-08T00:00:01+00:00"
                ),
                containment_removed_at="2026-08-08T00:00:02+00:00",
            ),
        }
        for label, mutate in mutations.items():
            changed = copy.deepcopy(workspace)
            mutate(changed["pending_process"])
            with self.subTest(label=label), self.assertRaises(RamdiskError):
                state_support._validate_benchmark_workspace(
                    changed,
                    manifest=self.fixture.manifest,
                    state_root=lambda: str(self.fixture.state),
                )

        active = copy.deepcopy(self.fixture.manifest)
        active["processes"] = [{"pid": 999}]
        with self.assertRaisesRegex(RamdiskError, "managed engines"):
            state_support._validate_benchmark_workspace(
                copy.deepcopy(workspace),
                manifest=active,
                state_root=lambda: str(self.fixture.state),
            )

    def test_inconclusive_proc_identity_retains_authority_and_workspace(self):
        manager = self.fixture.manager(
            process_supervisor=self.supervisor,
            process_starttime=lambda _pid: (_ for _ in ()).throw(
                PermissionError("proc denied")
            ),
        )
        roots = manager._prepare(
            self.fixture.manifest, {"protocol_id": "a" * 64}, None
        )
        record = self._pending_record(roots, "gate-released")
        self.fixture.manifest["benchmark_workspace"]["pending_process"] = record
        self.supervisor.containment = copy.deepcopy(record["containment"])
        self.supervisor.leaf_exists = True

        with self.assertRaisesRegex(
            benchmark.WorkspaceCleanupError,
            "inconclusive",
        ):
            manager.recover(self.fixture.manifest)
        self.assertIn(
            "pending_process",
            self.fixture.manifest["benchmark_workspace"],
        )
        self.assertFalse(self.supervisor.removed)
        self.assertFalse(
            any(event[0] == "umount" for event in self.fixture.events)
        )

    def test_malformed_existing_pending_process_fails_as_cleanup_error(self):
        manager = self.fixture.manager(
            process_supervisor=self.supervisor,
            process_starttime=self._starttime,
        )
        roots = manager._prepare(
            self.fixture.manifest, {"protocol_id": "a" * 64}, None
        )
        self.fixture.manifest["benchmark_workspace"]["pending_process"] = {
            "version": 1,
            "operation_id": "replicate:" + "b" * 32,
        }
        with self.assertRaisesRegex(
            benchmark._EngineCleanupError,
            "cannot enter supervised",
        ):
            with manager.process_attempt(
                recheck=lambda: None,
                **self._metadata(roots)
            ):
                self.fail("malformed pending authority must prevent launch")
        self.assertIsNone(manager._active_process_attempt)

    def test_removed_marker_requires_the_exact_leaf_path_to_be_absent(self):
        for replacement in (False, True):
            with self.subTest(replacement=replacement):
                root = Path(self.temporary_root) / (
                    "replacement" if replacement else "leaked"
                )
                root.mkdir()
                fixture = WorkspaceFixture(str(root))
                supervisor = FakeProcessSupervisor(fixture)

                def starttime(_pid):
                    raise FileNotFoundError("process is absent")

                manager = fixture.manager(
                    process_supervisor=supervisor,
                    process_starttime=starttime,
                )
                roots = manager._prepare(
                    fixture.manifest, {"protocol_id": "a" * 64}, None
                )
                original_fixture = self.fixture
                original_supervisor = self.supervisor
                try:
                    self.fixture = fixture
                    self.supervisor = supervisor
                    record = self._pending_record(roots, "gate-released")
                finally:
                    self.fixture = original_fixture
                    self.supervisor = original_supervisor
                record.update(
                    containment_removal_authorized_at=(
                        "2026-08-08T00:00:01+00:00"
                    ),
                    containment_removed_at="2026-08-08T00:00:02+00:00",
                )
                fixture.manifest["benchmark_workspace"][
                    "pending_process"
                ] = record
                supervisor.containment = copy.deepcopy(record["containment"])
                supervisor.leaf_exists = True
                supervisor.replacement = replacement

                with self.assertRaisesRegex(
                    benchmark.WorkspaceCleanupError,
                    "remains present|identity changed",
                ):
                    manager._recover_pending_process(fixture.manifest)
                self.assertIn(
                    "pending_process",
                    fixture.manifest["benchmark_workspace"],
                )

    @staticmethod
    def _metadata_for_fixture(fixture, roots):
        protocol_id = "a" * 64
        treatment_id = "tmpfs-rammap-local"
        launch_id = "b" * 32
        state_dir = (
            fixture.state
            / "causal-benchmark-state"
            / protocol_id
            / ("0000-003-%s-%s" % (treatment_id, launch_id))
        )
        state_dir.mkdir(parents=True, exist_ok=True)
        return {
            "protocol_id": protocol_id,
            "treatment_id": treatment_id,
            "block_index": 0,
            "sequence": 3,
            "launch_id": launch_id,
            "state_dir": str(state_dir),
            "weights_dir": roots["local"]["path"],
        }

class WorkspaceBindingTest(unittest.TestCase):
    def test_workspace_binding_preserves_identity_and_rejects_alias_mounts(self):
        with canonical_temporary_directory() as directory:
            first = Path(directory) / "first"
            second = Path(directory) / "second"
            first.mkdir()
            second.mkdir()
            shared_identity = {"mount_id": 10, "device": "1:2"}
            roots = {
                "interleaved": {
                    "name": "interleaved",
                    "role": "deployment",
                    "operation_id": "deployment:mount:0",
                    "path": str(first),
                    "mode": "interleave",
                    "nodes": [0, 1],
                    "node": None,
                    "size_bytes": 1024,
                    "policy": "interleave=static:0-1",
                    "requested": {"source": "tmpfs"},
                    "source_fingerprint": "a" * 64,
                    "verified": True,
                    "identity": shared_identity,
                },
                "local": {
                    "name": "local",
                    "role": "scratch",
                    "operation_id": "benchmark:" + "a" * 32,
                    "path": str(second),
                    "mode": "local",
                    "nodes": [0],
                    "node": 0,
                    "size_bytes": 1024,
                    "policy": "bind=static:0",
                    "requested": {"source": "tmpfs"},
                    "source_fingerprint": "a" * 64,
                    "verified": True,
                    "identity": shared_identity,
                },
            }

            with self.assertRaisesRegex(
                RamdiskError, "distinct devices and mount identities"
            ):
                benchmark._validate_workspace_roots(
                    type("Manager", (), {"verify": lambda _self, _value: True})(),
                    roots,
                )


if __name__ == "__main__":
    unittest.main()
