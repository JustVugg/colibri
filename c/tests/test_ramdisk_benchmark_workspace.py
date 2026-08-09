"""Durable dual-root workspace recovery for causal RAMMAP experiments."""

import copy
import contextlib
import os
import sys
import tempfile
import unittest
from pathlib import Path


C_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(C_DIR))

from ramdisk_support import benchmark  # noqa: E402
from ramdisk_support import state as state_support  # noqa: E402
from ramdisk_support.common import RamdiskError  # noqa: E402


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

    def manager(self, *, available_bytes=1 << 40):
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
        )


class DurableWorkspaceTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.fixture = WorkspaceFixture(self.temporary.name)

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

        missing_deployment = copy.deepcopy(self.fixture.manifest)
        missing_deployment["deployment_id"] = None
        with self.assertRaisesRegex(RamdiskError, "invalid benchmark workspace"):
            state_support._validate_benchmark_workspace(
                copy.deepcopy(self.fixture.manifest["benchmark_workspace"]),
                manifest=missing_deployment,
                state_root=lambda: str(self.fixture.state),
            )


class WorkspaceBindingTest(unittest.TestCase):
    def test_workspace_binding_preserves_identity_and_rejects_alias_mounts(self):
        with tempfile.TemporaryDirectory() as directory:
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
