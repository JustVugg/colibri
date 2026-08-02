"""RAM-disk durable state, safety, and lifecycle tests."""

import copy

if __package__:
    from .ramdisk_test_support import *  # noqa: F401,F403
else:
    from ramdisk_test_support import *  # noqa: F401,F403

from ramdisk_support import lifecycle as lifecycle_support
from ramdisk_support import linux_ops
from ramdisk_support import state as state_support


class StateAndSafetyTest(unittest.TestCase):
    FINGERPRINT = "sha256:" + ("a" * 64)
    GLM_ENGINE_ID = 3815245270
    USAGE_HEADER = "-1 1 2\n-2 1 %d\n" % GLM_ENGINE_ID

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = str(Path(self.temp.name).resolve())
        self.env = mock.patch.dict(
            os.environ,
            {
                "XDG_STATE_HOME": os.path.join(self.root, "state"),
                "COLI_RAMDISK_MANIFEST": os.path.join(self.root, "manifest.json"),
            },
        )
        self.env.start()
        self.filesystem = mock.patch.object(ramdisk, "_filesystem_for_path", return_value="ext4")
        self.filesystem.start()

    def tearDown(self):
        self.filesystem.stop()
        self.env.stop()
        self.temp.cleanup()

    def manifest(self, state="ready", mount_paths=None, processes=None):
        """Return a schema-valid lifecycle record for focused safety tests."""
        mount_paths = mount_paths or ["/mnt/colibri-test"]
        processes = processes or []
        topology = "per-node" if len(mount_paths) > 1 else "interleaved"
        mount_root = mount_paths[0] if topology == "interleaved" else os.path.dirname(mount_paths[0])
        nodes = list(range(len(mount_paths))) if topology == "per-node" else [0]
        planned = [
            {"path": path, "node": nodes[index] if topology == "per-node" else None}
            for index, path in enumerate(mount_paths)
        ]
        mounted = [
            {
                "path": path,
                "node": planned[index]["node"],
                "identity": {
                    "mount_id": index + 4,
                    "device": "0:%d" % (index + 9),
                },
            }
            for index, path in enumerate(mount_paths)
        ]
        fingerprint_dir = self.FINGERPRINT.split(":", 1)[1]
        complete_processes = []
        for index, partial in enumerate(processes):
            record = dict(partial)
            node = planned[index]["node"]
            port = 8000 + index
            label = "interleaved" if node is None else "node-%d" % node
            record.update(
                {
                    "pgid": record["pid"],
                    "uid": host_uid(),
                    "starttime": 100 + record["pid"],
                    "nonce": "%048x" % (index + 1),
                    "port": port,
                    "node": node,
                    "weights_dir": planned[index]["path"],
                    "state_dir": os.path.join(
                        ramdisk._state_root(), "engines", fingerprint_dir, label
                    ),
                    "usage_baseline": {},
                    "command": [
                        str(C_DIR / "coli"),
                        "serve",
                        "--model",
                        os.path.join(self.root, "model"),
                        "--port",
                        str(port),
                    ],
                }
            )
            complete_processes.append(record)
        return {
            "version": ramdisk.MANIFEST_VERSION,
            "state": state,
            "model_fingerprint": self.FINGERPRINT,
            "plan": {
                "topology": topology,
                "mount_root": mount_root,
                "mounts": planned,
                "hardware": hardware_fixture(nodes=len(mount_paths) if topology == "per-node" else 1),
                "model": {
                    "path": os.path.join(self.root, "model"),
                    "fingerprint": self.FINGERPRINT,
                },
                "durable_state": {
                    "root": ramdisk._state_root(),
                    "manifest": ramdisk._manifest_path(),
                    "benchmarks": ramdisk._benchmarks_path(),
                },
                "source_shards": [{"name": "model.safetensors"}],
            },
            "mounts": mounted,
            "processes": complete_processes,
        }

    def recovery_state_dir(self, node=None):
        label = "interleaved" if node is None else "node-%d" % node
        return os.path.join(
            ramdisk._state_root(),
            "engines",
            self.FINGERPRINT.split(":", 1)[1],
            label,
        )

    def test_usage_delta_merge_and_crash_recovery_are_idempotent(self):
        model_dir = os.path.join(self.root, "model")
        canonical = os.path.join(model_dir, ".coli_usage")
        state_dir = os.path.join(self.root, "node-state")
        os.makedirs(model_dir, mode=0o750)
        os.makedirs(state_dir, mode=0o700)
        ramdisk._usage_write(canonical, {"0:1": 10, "0:2": 3})
        ramdisk._usage_write(os.path.join(state_dir, ".coli_usage"), {"0:1": 14, "0:2": 3, "0:3": 2})
        record = {"state_dir": state_dir, "usage_baseline": {"0:1": 10, "0:2": 3}}
        ramdisk._merge_usage(record, canonical)
        self.assertEqual(ramdisk._usage_read(canonical), {"0:1": 14, "0:2": 3, "0:3": 2})
        ramdisk._merge_usage(record, canonical)
        self.assertEqual(ramdisk._usage_read(canonical), {"0:1": 14, "0:2": 3, "0:3": 2})

        merge_id = "a" * 32
        delta_path = os.path.join(state_dir, ".coli_usage.delta.json")
        ramdisk._usage_write(canonical, {"0:1": 15}, merge_id=merge_id)
        ramdisk._atomic_json(delta_path, {"version": 1, "id": merge_id, "delta": {"0:1": 1}})
        ramdisk._recover_delta(state_dir, canonical)
        self.assertEqual(ramdisk._usage_read(canonical)["0:1"], 15)
        self.assertFalse(os.path.exists(delta_path))
        self.assertEqual(os.stat(model_dir).st_mode & 0o777, 0o750)

    def test_usage_history_read_is_optional_only_when_absent(self):
        missing = os.path.join(self.root, "missing-usage")
        self.assertEqual(state_support._usage_read(missing), {})

        denied = os.path.join(self.root, "denied-usage")
        with mock.patch(
            "builtins.open",
            side_effect=PermissionError("permission denied"),
        ):
            with self.assertRaisesRegex(
                ramdisk.RamdiskError,
                "cannot read usage state.*permission denied",
            ):
                state_support._usage_read(denied)

        invalid = os.path.join(self.root, "invalid-utf8-usage")
        with open(invalid, "wb") as stream:
            stream.write(b"0 1 7\n\xff")
        with self.assertRaisesRegex(
            ramdisk.RamdiskError,
            "cannot read usage state",
        ):
            state_support._usage_read(invalid)

    def test_headered_usage_copy_merge_and_recovery_preserve_metadata(self):
        model_dir = os.path.join(self.root, "model")
        canonical = os.path.join(model_dir, ".coli_usage")
        state_dir = os.path.join(self.root, "node-state")
        os.makedirs(model_dir)
        os.makedirs(state_dir)
        Path(canonical).write_text(
            self.USAGE_HEADER + "0 1 10\n",
            encoding="utf-8",
        )

        baseline = ramdisk._usage_read(canonical)
        self.assertEqual(baseline["-1:1"], 2)
        self.assertEqual(baseline["-2:1"], self.GLM_ENGINE_ID)
        state_usage = os.path.join(state_dir, ".coli_usage")
        ramdisk._usage_write(state_usage, baseline)
        self.assertTrue(
            Path(state_usage).read_text(encoding="utf-8").startswith(
                self.USAGE_HEADER
            )
        )

        current = dict(baseline)
        current["0:1"] = 13
        ramdisk._usage_write(state_usage, current)
        record = {
            "state_dir": state_dir,
            "usage_baseline": baseline,
            "usage_merge_id": "b" * 32,
        }
        ramdisk._merge_usage(record, canonical)
        merged = ramdisk._usage_read(canonical)
        self.assertEqual(merged["0:1"], 13)
        self.assertEqual(merged["-1:1"], 2)
        self.assertEqual(merged["-2:1"], self.GLM_ENGINE_ID)

        recovery_dir = os.path.join(self.root, "recovery-state")
        os.makedirs(recovery_dir)
        recovery_id = "c" * 32
        ramdisk._atomic_json(
            os.path.join(recovery_dir, ".coli_usage.delta.json"),
            {
                "version": 1,
                "id": recovery_id,
                "delta": {"0:1": 2},
                "headers": {
                    "-1:1": 2,
                    "-2:1": self.GLM_ENGINE_ID,
                },
            },
        )
        ramdisk._recover_delta(recovery_dir, canonical)
        recovered = ramdisk._usage_read(canonical)
        self.assertEqual(recovered["0:1"], 15)
        self.assertEqual(recovered["-1:1"], 2)
        self.assertEqual(recovered["-2:1"], self.GLM_ENGINE_ID)

    def test_usage_metadata_must_be_complete_and_compatible(self):
        model_dir = os.path.join(self.root, "model")
        canonical = os.path.join(model_dir, ".coli_usage")
        state_dir = os.path.join(self.root, "node-state")
        os.makedirs(model_dir)
        os.makedirs(state_dir)
        Path(canonical).write_text(
            self.USAGE_HEADER + "0 1 10\n",
            encoding="utf-8",
        )
        baseline = ramdisk._usage_read(canonical)
        state_usage = os.path.join(state_dir, ".coli_usage")
        Path(state_usage).write_text(
            "-1 1 2\n-2 1 1\n0 1 12\n",
            encoding="utf-8",
        )
        record = {"state_dir": state_dir, "usage_baseline": baseline}
        with self.assertRaisesRegex(ramdisk.RamdiskError, "engine"):
            ramdisk._merge_usage(record, canonical)
        self.assertEqual(ramdisk._usage_read(canonical), baseline)

        incomplete = os.path.join(self.root, "incomplete.coli_usage")
        Path(incomplete).write_text("-1 1 2\n0 1 3\n", encoding="utf-8")
        with self.assertRaisesRegex(ramdisk.RamdiskError, "both"):
            ramdisk._usage_read(incomplete)

    def test_legacy_seed_upgrades_to_engine_header_and_old_journal_recovers(self):
        model_dir = os.path.join(self.root, "model")
        canonical = os.path.join(model_dir, ".coli_usage")
        state_dir = os.path.join(self.root, "node-state")
        os.makedirs(model_dir)
        os.makedirs(state_dir)
        Path(canonical).write_text("0 1 10\n", encoding="utf-8")
        baseline = ramdisk._usage_read(canonical)
        Path(state_dir, ".coli_usage").write_text(
            self.USAGE_HEADER + "0 1 10\n",
            encoding="utf-8",
        )
        ramdisk._merge_usage(
            {"state_dir": state_dir, "usage_baseline": baseline},
            canonical,
        )
        upgraded = ramdisk._usage_read(canonical)
        self.assertEqual(upgraded["0:1"], 10)
        self.assertEqual(upgraded["-1:1"], 2)
        self.assertEqual(upgraded["-2:1"], self.GLM_ENGINE_ID)

        current = dict(upgraded)
        current["0:1"] = 12
        ramdisk._usage_write(
            os.path.join(state_dir, ".coli_usage"),
            current,
        )
        ramdisk._merge_usage(
            {
                "state_dir": state_dir,
                "usage_baseline": baseline,
                "usage_merge_id": "e" * 32,
            },
            canonical,
        )

        # Journals created by the pre-header RAM-disk manager have no metadata.
        # They remain valid legacy deltas, while the canonical identity wins.
        recovery_dir = os.path.join(self.root, "old-journal")
        os.makedirs(recovery_dir)
        ramdisk._atomic_json(
            os.path.join(recovery_dir, ".coli_usage.delta.json"),
            {
                "version": 1,
                "id": "d" * 32,
                "delta": {"0:1": 1},
            },
        )
        ramdisk._recover_delta(recovery_dir, canonical)
        recovered = ramdisk._usage_read(canonical)
        self.assertEqual(recovered["0:1"], 13)
        self.assertEqual(recovered["-1:1"], 2)
        self.assertEqual(recovered["-2:1"], self.GLM_ENGINE_ID)

    def test_atomic_json_never_chmods_an_existing_override_parent(self):
        parent = os.path.join(self.root, "shared-parent")
        os.mkdir(parent, 0o755)
        before = os.stat(parent).st_mode & 0o777
        ramdisk._atomic_json(os.path.join(parent, "manifest.json"), {"ok": True})
        self.assertEqual(os.stat(parent).st_mode & 0o777, before)

    def test_private_state_directory_rejects_existing_symlink_without_chmod(self):
        target = os.path.join(self.root, "redirect-target")
        link = os.path.join(self.root, "redirect-link")
        os.mkdir(target, 0o755)
        os.symlink(target, link)
        before = os.stat(target).st_mode & 0o777
        with self.assertRaisesRegex(ramdisk.RamdiskError, "contains a symlink"):
            ramdisk._ensure_private_dir(link)
        self.assertEqual(os.stat(target).st_mode & 0o777, before)

    def test_derived_state_directory_must_remain_on_durable_filesystem(self):
        state_dir = os.path.join(self.root, "engine-state")
        os.mkdir(state_dir)
        with mock.patch.object(ramdisk, "_filesystem_for_path", return_value="tmpfs"):
            with self.assertRaisesRegex(ramdisk.RamdiskError, "volatile filesystem"):
                ramdisk._assert_durable_state_dir(state_dir, plan=self.manifest()["plan"])

    def test_two_node_usage_markers_survive_crash_between_manifest_saves(self):
        model_dir = os.path.join(self.root, "model")
        canonical = os.path.join(model_dir, ".coli_usage")
        os.makedirs(model_dir)
        ramdisk._usage_write(canonical, {"0:1": 10})
        records = []
        for index, value in enumerate((12, 13), 1):
            state_dir = os.path.join(self.root, "node-%d" % index)
            os.makedirs(state_dir)
            ramdisk._usage_write(os.path.join(state_dir, ".coli_usage"), {"0:1": value})
            record = {
                "state_dir": state_dir,
                "usage_baseline": {"0:1": 10},
                "usage_merge_id": ("%x" % index) * 32,
            }
            ramdisk._merge_usage(record, canonical)
            records.append(record)
        self.assertEqual(ramdisk._usage_read(canonical), {"0:1": 15})
        self.assertEqual(ramdisk._usage_merge_ids(canonical), {"1" * 32, "2" * 32})

        # Simulate both records still looking uncommitted after a manager crash.
        for index, record in enumerate(records, 1):
            ramdisk._atomic_json(
                os.path.join(record["state_dir"], ".coli_usage.delta.json"),
                {"version": 1, "id": record["usage_merge_id"], "delta": {"0:1": index + 1}},
            )
            ramdisk._recover_delta(record["state_dir"], canonical)
        self.assertEqual(ramdisk._usage_read(canonical), {"0:1": 15})

    def test_process_identity_rejects_uid_starttime_and_nonce_mismatch(self):
        record = {"pid": 44, "uid": 1000, "starttime": 99, "nonce": "expected"}
        with mock.patch.object(
            ramdisk,
            "_proc_identity",
            return_value={"pid": 44, "uid": 1000, "starttime": 99, "nonce": "other", "pgid": 44},
        ):
            matches, reason, _ = ramdisk._process_matches(record)
        self.assertFalse(matches)
        self.assertEqual(reason, "foreign-nonce")

    @requires_linux_operational
    def test_manifest_rejects_missing_nonce_before_process_signaling(self):
        manifest = self.manifest(
            state="running", processes=[{"pid": 12345}]
        )
        manifest["processes"][0].pop("nonce")
        ramdisk._atomic_json(ramdisk._manifest_path(), manifest)
        with self.assertRaisesRegex(ramdisk.RamdiskError, "unsafe managed process"):
            ramdisk._load_manifest(required=True)

    def test_manifest_rejects_corrupt_process_accounting_metadata(self):
        manifest = self.manifest(
            state="running",
            processes=[{"pid": 12346}],
        )
        manifest["processes"][0]["usage_baseline"] = {"0:1": True}
        manifest["processes"][0]["usage_merge_id"] = "UPPERCASE"
        ramdisk._atomic_json(ramdisk._manifest_path(), manifest)

        with self.assertRaisesRegex(
            ramdisk.RamdiskError,
            "unsafe managed process",
        ):
            ramdisk._load_manifest(required=True)

    def test_manifest_rejects_recovery_state_dir_outside_exact_node_path(self):
        manifest = self.manifest(state="error")
        manifest["recovery"] = {
            "operation": "start",
            "state": "attention-required",
            "retained_processes": [
                {
                    "pid": 12347,
                    "pgid": 12347,
                    "node": None,
                    "state_dir": os.path.join(self.root, "arbitrary-state"),
                    "usage_baseline": {},
                    "usage_merge_id": "a" * 32,
                    "error": "absence unproven",
                }
            ],
        }
        ramdisk._atomic_json(ramdisk._manifest_path(), manifest)

        with self.assertRaisesRegex(
            ramdisk.RamdiskError,
            "unsafe or duplicate state directory",
        ):
            ramdisk._load_manifest(required=True)

    def test_manifest_rejects_mount_layout_outside_v1_root(self):
        manifest = self.manifest()
        manifest["plan"]["mount_root"] = os.path.join(self.root, "mount")
        manifest["plan"]["mounts"][0]["path"] = manifest["plan"]["mount_root"]
        manifest["mounts"][0]["path"] = manifest["plan"]["mount_root"]
        ramdisk._atomic_json(ramdisk._manifest_path(), manifest)
        with self.assertRaisesRegex(ramdisk.RamdiskError, "unsafe mount root"):
            ramdisk._load_manifest(required=True)

    def test_error_manifest_accepts_pending_mount_without_an_identity(self):
        manifest = self.manifest(state="error")
        manifest["mounts"][0].pop("identity")
        manifest["mounts"][0]["ownership"] = "pending"
        manifest["error"] = "mount helper outcome is unknown"
        ramdisk._atomic_json(ramdisk._manifest_path(), manifest)

        loaded = ramdisk._load_manifest(required=True)

        self.assertEqual(loaded["state"], "error")
        self.assertEqual(loaded["mounts"][0]["ownership"], "pending")
        self.assertNotIn("identity", loaded["mounts"][0])

    def test_ready_manifest_rejects_pending_mount_ownership(self):
        manifest = self.manifest(state="ready")
        manifest["mounts"][0].pop("identity")
        manifest["mounts"][0]["ownership"] = "pending"
        ramdisk._atomic_json(ramdisk._manifest_path(), manifest)

        with self.assertRaisesRegex(ramdisk.RamdiskError, "pending mount"):
            ramdisk._load_manifest(required=True)

    def test_manifest_rejects_retained_process_recovery_outside_error_state(self):
        manifest = self.manifest(state="stopped")
        manifest["recovery"] = {
            "operation": "start",
            "state": "attention-required",
            "retained_processes": [
                {
                    "pid": 12001,
                    "pgid": 12001,
                    "node": None,
                    "state_dir": self.recovery_state_dir(),
                    "usage_baseline": {"0:1": 7},
                    "usage_merge_id": "1" * 32,
                    "error": "group absence unproven",
                }
            ],
        }
        ramdisk._atomic_json(ramdisk._manifest_path(), manifest)

        with self.assertRaisesRegex(
            ramdisk.RamdiskError,
            "retained process recovery.*error state",
        ):
            ramdisk._load_manifest(required=True)

    def test_outcome_unknown_pending_launch_blocks_all_mutating_actions(self):
        pending = {
            "operation_id": "start:" + ("6" * 32),
            "nonce": "7" * 48,
            "port": 8000,
            "node": None,
            "state_dir": self.recovery_state_dir(),
            "usage_baseline": {"0:1": 3},
            "usage_merge_id": "6" * 32,
        }

        def write_pending():
            manifest = self.manifest(state="starting")
            manifest["pending_launches"] = [copy.deepcopy(pending)]
            ramdisk._atomic_json(ramdisk._manifest_path(), manifest)

        write_pending()
        with mock.patch.object(ramdisk.subprocess, "Popen") as popen:
            with self.assertRaisesRegex(
                ramdisk.RamdiskError,
                "pre-spawn managed launch has an unknown outcome",
            ):
                ramdisk.start.__wrapped__(
                    argparse.Namespace(base_port=None),
                    cli_path=sys.executable,
                )
        popen.assert_not_called()

        write_pending()
        with mock.patch.object(
            ramdisk, "_terminate_verified_group"
        ) as terminate, mock.patch.object(
            ramdisk, "_merge_usage"
        ) as merge:
            with self.assertRaisesRegex(
                ramdisk.RamdiskError,
                "pre-spawn managed launch has an unknown outcome",
            ):
                ramdisk.stop()
        terminate.assert_not_called()
        merge.assert_not_called()
        stopped_refusal = ramdisk._load_manifest(required=True)
        self.assertEqual(stopped_refusal["state"], "error")
        self.assertEqual(stopped_refusal["pending_launches"], [pending])

        write_pending()
        with mock.patch.object(ramdisk, "_umount_path") as unmount, mock.patch.object(
            ramdisk, "_durable_unlink"
        ) as unlink:
            with self.assertRaisesRegex(
                ramdisk.RamdiskError,
                "pre-spawn managed launch has an unknown outcome",
            ):
                ramdisk.destroy(argparse.Namespace(yes=True))
        unmount.assert_not_called()
        unlink.assert_not_called()

    def test_start_refuses_unresolved_unpublished_process_recovery(self):
        manifest = self.manifest(state="error")
        manifest["recovery"] = {
            "operation": "start",
            "state": "attention-required",
            "retained_processes": [
                {
                    "pid": 12002,
                    "pgid": 12002,
                    "node": None,
                    "state_dir": self.recovery_state_dir(),
                    "usage_baseline": {"0:1": 7},
                    "usage_merge_id": "2" * 32,
                    "error": "group absence unproven",
                }
            ],
        }
        ramdisk._atomic_json(ramdisk._manifest_path(), manifest)

        with mock.patch.object(ramdisk.subprocess, "Popen") as popen:
            with self.assertRaisesRegex(
                ramdisk.RamdiskError,
                "unpublished managed-child absence is unproven",
            ):
                ramdisk.start.__wrapped__(
                    argparse.Namespace(base_port=None),
                    cli_path=sys.executable,
                )

        popen.assert_not_called()

    def test_stop_refuses_unresolved_unpublished_process_recovery(self):
        manifest = self.manifest(state="error")
        manifest["recovery"] = {
            "operation": "start",
            "state": "attention-required",
            "retained_processes": [
                {
                    "pid": 12003,
                    "pgid": 12003,
                    "node": None,
                    "state_dir": self.recovery_state_dir(),
                    "usage_baseline": {"0:1": 7},
                    "usage_merge_id": "3" * 32,
                    "error": "group absence unproven",
                }
            ],
        }
        ramdisk._atomic_json(ramdisk._manifest_path(), manifest)

        with mock.patch.object(
            ramdisk, "_terminate_verified_group"
        ) as terminate, mock.patch.object(
            ramdisk, "_group_alive", return_value=True
        ), mock.patch.object(
            ramdisk, "_merge_usage"
        ) as merge:
            with self.assertRaisesRegex(
                ramdisk.RamdiskError,
                "unpublished.*unproven",
            ):
                ramdisk.stop()

        terminate.assert_not_called()
        merge.assert_not_called()
        persisted = ramdisk._load_manifest(required=True)
        self.assertEqual(persisted["state"], "error")
        self.assertEqual(
            persisted["recovery"]["retained_processes"][0]["pid"],
            12003,
        )

    def test_stop_reconciles_unpublished_usage_once_with_exact_baseline(self):
        merge_id = "5" * 32
        model_dir = os.path.join(self.root, "accounting-model")
        state_dir = os.path.join(self.root, "accounting-state")
        os.makedirs(model_dir)
        os.makedirs(state_dir)
        canonical = os.path.join(model_dir, ".coli_usage")
        state_support._usage_write(canonical, {"0:1": 10})
        state_support._usage_write(
            os.path.join(state_dir, ".coli_usage"),
            {"0:1": 12},
        )
        entry = {
            "pid": 12005,
            "pgid": 12005,
            "node": None,
            "state_dir": state_dir,
            "usage_baseline": {"0:1": 10},
            "usage_merge_id": merge_id,
            "error": "group absence unproven",
        }
        manifest = {
            "state": "error",
            "plan": {
                "model": {"path": model_dir},
                "mounts": [],
            },
            "recovery": {
                "operation": "start",
                "state": "attention-required",
                "retained_processes": [entry],
            },
        }

        def merge(record, canonical_path, plan=None):
            self.assertEqual(record["usage_baseline"], {"0:1": 10})
            self.assertEqual(record["usage_merge_id"], merge_id)
            return state_support._merge_usage(
                record,
                canonical_path,
                plan=plan,
                filesystem_for_path=lambda ignored: "ext4",
                source_still_matches=lambda ignored: None,
            )

        saves = {"count": 0}

        def fail_first_post_merge_save(_manifest):
            saves["count"] += 1
            if saves["count"] == 1:
                raise OSError("manager lost manifest write after merge")

        with self.assertRaisesRegex(
            ramdisk.RamdiskError,
            "manager lost manifest write after merge",
        ):
            lifecycle_support._reconcile_unpublished_processes(
                manifest,
                group_alive=lambda ignored: False,
                merge_usage=merge,
                save_manifest=fail_first_post_merge_save,
            )
        self.assertEqual(state_support._usage_read(canonical), {"0:1": 12})

        # A fresh manager reloads the still-retained transaction and retries
        # the same stable id. The canonical marker makes that retry a no-op.
        restarted = copy.deepcopy(manifest)
        lifecycle_support._reconcile_unpublished_processes(
            restarted,
            group_alive=lambda ignored: False,
            merge_usage=merge,
            save_manifest=lambda ignored: None,
        )

        self.assertEqual(state_support._usage_read(canonical), {"0:1": 12})
        self.assertEqual(
            restarted["recovery"]["retained_processes"],
            [],
        )

    def test_destroy_refuses_unresolved_unpublished_process_recovery(self):
        manifest = self.manifest(state="error")
        manifest["recovery"] = {
            "operation": "start",
            "state": "attention-required",
            "retained_processes": [
                {
                    "pid": 12004,
                    "pgid": 12004,
                    "node": None,
                    "state_dir": self.recovery_state_dir(),
                    "usage_baseline": {"0:1": 7},
                    "usage_merge_id": "4" * 32,
                    "error": "group absence unproven",
                }
            ],
        }
        ramdisk._atomic_json(ramdisk._manifest_path(), manifest)

        with mock.patch.object(
            ramdisk, "_group_alive", return_value=True
        ), mock.patch.object(ramdisk, "_umount_path") as unmount, mock.patch.object(
            ramdisk, "_durable_unlink"
        ) as unlink:
            with self.assertRaisesRegex(
                ramdisk.RamdiskError,
                "unpublished.*unproven",
            ):
                ramdisk.destroy(argparse.Namespace(yes=True))

        unmount.assert_not_called()
        unlink.assert_not_called()

    @requires_linux_operational
    def test_stop_validates_every_pid_before_signaling_any(self):
        manifest = self.manifest(
            state="running",
            mount_paths=["/mnt/colibri-test/node0", "/mnt/colibri-test/node1"],
            processes=[{"pid": 1}, {"pid": 2}],
        )
        ramdisk._atomic_json(ramdisk._manifest_path(), manifest)

        def match(record):
            if record["pid"] == 1:
                return True, "running", {"pgid": 1}
            return False, "foreign-uid", {"pgid": 2}

        with mock.patch.object(ramdisk, "_process_matches", side_effect=match), mock.patch.object(
            os, "killpg"
        ) as kill:
            with self.assertRaisesRegex(ramdisk.RamdiskError, "unverified"):
                ramdisk.stop()
        kill.assert_not_called()

    def test_stop_persists_procfs_preflight_failure_before_any_signal(self):
        manifest = self.manifest(
            state="running",
            processes=[{"pid": 12340}],
        )
        ramdisk._atomic_json(ramdisk._manifest_path(), manifest)

        with mock.patch.object(
            ramdisk,
            "_process_matches",
            side_effect=ramdisk.RamdiskError("procfs enumeration unreadable"),
        ), mock.patch.object(
            ramdisk, "_terminate_verified_group"
        ) as terminate, mock.patch.object(
            ramdisk, "_merge_usage"
        ) as merge:
            with self.assertRaisesRegex(
                ramdisk.RamdiskError,
                "unverified.*procfs enumeration unreadable",
            ):
                ramdisk.stop()

        terminate.assert_not_called()
        merge.assert_not_called()
        persisted = ramdisk._load_manifest(required=True)
        self.assertEqual(persisted["state"], "error")
        self.assertIn(
            "procfs enumeration unreadable",
            persisted["processes"][0]["stop_error"],
        )
        self.assertNotIn("stopped_at", persisted["processes"][0])

    def test_stop_persists_post_termination_revalidation_failure(self):
        manifest = self.manifest(
            state="running",
            processes=[{"pid": 12341}],
        )
        ramdisk._atomic_json(ramdisk._manifest_path(), manifest)
        running = (True, "running", {"pid": 12341, "pgid": 12341})

        with mock.patch.object(
            ramdisk,
            "_process_matches",
            side_effect=(
                running,
                ramdisk.RamdiskError("post-termination procfs unreadable"),
            ),
        ), mock.patch.object(
            ramdisk,
            "_terminate_verified_group",
            side_effect=ramdisk.RamdiskError(
                "termination revalidation unreadable"
            ),
        ), mock.patch.object(
            ramdisk, "_merge_usage"
        ) as merge:
            with self.assertRaisesRegex(
                ramdisk.RamdiskError,
                "termination revalidation unreadable",
            ):
                ramdisk.stop()

        merge.assert_not_called()
        persisted = ramdisk._load_manifest(required=True)
        self.assertEqual(persisted["state"], "error")
        self.assertIn(
            "post-termination procfs unreadable",
            persisted["processes"][0]["stop_error"],
        )
        self.assertNotIn("stopped_at", persisted["processes"][0])

    def test_stop_refuses_nonmanaged_mount_recovery(self):
        for ownership in ("pending", "identified"):
            with self.subTest(ownership=ownership):
                manifest = self.manifest(state="error")
                manifest["mounts"][0]["ownership"] = ownership
                if ownership == "pending":
                    manifest["mounts"][0].pop("identity")
                ramdisk._atomic_json(ramdisk._manifest_path(), manifest)

                with mock.patch.object(
                    ramdisk, "_terminate_verified_group"
                ) as terminate, mock.patch.object(
                    ramdisk, "_merge_usage"
                ) as merge:
                    with self.assertRaisesRegex(
                        ramdisk.RamdiskError,
                        "non-managed mount ownership",
                    ):
                        ramdisk.stop()

                terminate.assert_not_called()
                merge.assert_not_called()
                persisted = ramdisk._load_manifest(required=True)
                self.assertEqual(persisted["state"], "error")
                self.assertEqual(
                    persisted["mounts"][0]["ownership"],
                    ownership,
                )

    @requires_linux_operational
    def test_stop_revalidates_identity_before_escalating_to_sigkill(self):
        record = {
            "pid": 12345,
            "pgid": 12345,
            "uid": host_uid(),
            "starttime": 91,
            "nonce": "a" * 48,
        }
        running = (
            True,
            "running",
            {"pid": 12345, "pgid": 12345},
        )
        reused = (
            False,
            "foreign-starttime",
            {"pid": 12345, "pgid": 12345},
        )
        with mock.patch.object(
            ramdisk,
            "_process_matches",
            side_effect=(running, reused),
        ), mock.patch.object(os, "killpg", create=True) as kill:
            failure = ramdisk._terminate_verified_group(
                record,
                term_seconds=0,
                kill_seconds=0,
            )

        kill.assert_called_once_with(12345, signal.SIGTERM)
        self.assertIn("identity changed before SIGKILL", failure)

    @requires_linux_operational
    def test_verified_stop_reaps_a_locally_owned_zombie_before_escalation(self):
        record = {
            "pid": 12346,
            "pgid": 12346,
            "uid": host_uid(),
            "starttime": 92,
            "nonce": "b" * 48,
        }
        process = mock.Mock(pid=12346)
        process.poll.side_effect = (None, 0)
        running = (
            True,
            "running",
            {"pid": 12346, "pgid": 12346},
        )
        stopped = (False, "not-running", None)
        ramdisk._track_managed_child(process)
        self.addCleanup(ramdisk._forget_managed_child, process.pid)
        with mock.patch.object(
            ramdisk,
            "_process_matches",
            side_effect=(running, stopped),
        ), mock.patch.object(os, "killpg", create=True) as kill:
            failure = ramdisk._terminate_verified_group(
                record,
                term_seconds=0,
                kill_seconds=0,
            )

        self.assertIsNone(failure)
        kill.assert_called_once_with(12346, signal.SIGTERM)
        self.assertNotIn(12346, ramdisk._managed_children)

    @requires_linux_operational
    def test_verified_termination_treats_retained_live_child_as_independent_evidence(self):
        record = {
            "pid": 12347,
            "pgid": 12347,
            "uid": host_uid(),
            "starttime": 93,
            "nonce": "c" * 48,
        }
        process = mock.Mock(pid=12347, returncode=None)
        process.poll.return_value = None
        ramdisk._track_managed_child(process)
        self.addCleanup(ramdisk._forget_managed_child, process.pid)

        with mock.patch.object(
            ramdisk,
            "_process_matches",
            return_value=(False, "not-running", None),
        ), mock.patch.object(os, "killpg", create=True) as kill:
            failure = ramdisk._terminate_verified_group(
                record,
                term_seconds=0,
                kill_seconds=0,
            )

        kill.assert_not_called()
        self.assertIn("retained managed child is still live", failure)
        self.assertTrue(ramdisk._managed_child_liveness(process.pid))

    def test_missing_retained_handle_preserves_not_running_result(self):
        record = {
            "pid": 12350,
            "pgid": 12350,
            "uid": host_uid(),
            "starttime": 94,
            "nonce": "d" * 48,
        }
        self.assertIsNone(ramdisk._managed_child_liveness(record["pid"]))

        with mock.patch.object(
            ramdisk,
            "_process_matches",
            return_value=(False, "not-running", None),
        ), mock.patch.object(os, "killpg", create=True) as kill:
            failure = ramdisk._terminate_verified_group(
                record,
                term_seconds=0,
                kill_seconds=0,
            )

        self.assertIsNone(failure)
        kill.assert_not_called()

    @requires_linux_operational
    def test_stop_persists_error_when_usage_merge_fails(self):
        manifest = self.manifest(state="running", processes=[{"pid": 12345}])
        ramdisk._atomic_json(ramdisk._manifest_path(), manifest)
        with mock.patch.object(
            ramdisk, "_process_matches", return_value=(False, "not-running", None)
        ), mock.patch.object(
            ramdisk, "_merge_usage", side_effect=ramdisk.RamdiskError("disk unavailable")
        ):
            with self.assertRaisesRegex(ramdisk.RamdiskError, "cleanup is incomplete"):
                ramdisk.stop()
        persisted = ramdisk._read_json(ramdisk._manifest_path())
        self.assertEqual(persisted["state"], "error")
        self.assertIn("disk unavailable", persisted["processes"][0]["usage_merge_error"])

    def test_stop_retry_clears_post_merge_save_error_without_double_merge(self):
        manifest = self.manifest(
            state="running",
            processes=[{"pid": 12351}],
        )
        durable = {"manifest": None}
        saves = {"count": 0}
        merge = mock.Mock()

        def fail_first_completion_save(current):
            saves["count"] += 1
            if saves["count"] == 2:
                raise OSError("manifest write failed after usage merge")
            durable["manifest"] = copy.deepcopy(current)

        common = {
            "process_matches": lambda record: (
                False,
                "not-running",
                None,
            ),
            "group_alive": lambda pgid: False,
            "managed_child_liveness": lambda pid: False,
            "terminate_verified_group": mock.Mock(),
            "merge_usage": merge,
        }
        with self.assertRaisesRegex(
            ramdisk.RamdiskError,
            "manifest write failed after usage merge",
        ):
            lifecycle_support.stop(
                load_manifest=lambda required=True: manifest,
                save_manifest=fail_first_completion_save,
                **common,
            )

        restarted = copy.deepcopy(durable["manifest"])
        stopped = lifecycle_support.stop(
            load_manifest=lambda required=True: restarted,
            save_manifest=lambda current: durable.update(
                manifest=copy.deepcopy(current)
            ),
            **common,
        )

        self.assertEqual(merge.call_count, 1)
        self.assertEqual(stopped["state"], "stopped")
        self.assertIn("usage_merged_at", stopped["processes"][0])
        self.assertNotIn("usage_merge_error", stopped["processes"][0])

    @requires_linux_operational
    def test_stop_does_not_merge_when_retained_child_is_live(self):
        manifest = self.manifest(state="running", processes=[{"pid": 12348}])
        ramdisk._atomic_json(ramdisk._manifest_path(), manifest)
        process = mock.Mock(pid=12348, returncode=None)
        process.poll.return_value = None
        ramdisk._track_managed_child(process)
        self.addCleanup(ramdisk._forget_managed_child, process.pid)

        with mock.patch.object(
            ramdisk,
            "_process_matches",
            return_value=(False, "not-running", None),
        ), mock.patch.object(
            ramdisk, "_terminate_verified_group"
        ) as terminate, mock.patch.object(
            ramdisk, "_merge_usage"
        ) as merge:
            with self.assertRaisesRegex(
                ramdisk.RamdiskError,
                "retained-managed-child-live",
            ):
                ramdisk.stop()

        terminate.assert_not_called()
        merge.assert_not_called()
        persisted = ramdisk._load_manifest(required=True)
        self.assertEqual(persisted["state"], "error")
        self.assertNotIn("stopped_at", persisted["processes"][0])
        self.assertNotIn("usage_merged_at", persisted["processes"][0])

    @requires_linux_operational
    def test_stop_preserves_termination_failure_until_group_absence_is_proven(self):
        manifest = self.manifest(state="running", processes=[{"pid": 12349}])
        ramdisk._atomic_json(ramdisk._manifest_path(), manifest)
        running = (True, "running", {"pid": 12349, "pgid": 12349})
        inconclusive = (
            False,
            "unverified-process-group",
            {"pgid": 12349},
        )

        with mock.patch.object(
            ramdisk,
            "_process_matches",
            side_effect=(running, inconclusive),
        ), mock.patch.object(
            ramdisk,
            "_terminate_verified_group",
            return_value="identity changed after SIGTERM",
        ), mock.patch.object(
            ramdisk, "_merge_usage"
        ) as merge:
            with self.assertRaisesRegex(
                ramdisk.RamdiskError,
                "identity changed after SIGTERM",
            ):
                ramdisk.stop()

        merge.assert_not_called()
        persisted = ramdisk._load_manifest(required=True)
        self.assertEqual(persisted["state"], "error")
        self.assertIn(
            "identity changed after SIGTERM",
            persisted["processes"][0]["stop_error"],
        )
        self.assertNotIn("stopped_at", persisted["processes"][0])

    @requires_linux_operational
    def test_stop_preserves_recoverable_error_for_incomplete_mount_layout(self):
        manifest = self.manifest(
            state="error",
            mount_paths=["/mnt/colibri-test/node0", "/mnt/colibri-test/node1"],
        )
        manifest["mounts"].pop()
        ramdisk._atomic_json(ramdisk._manifest_path(), manifest)

        stopped = ramdisk.stop()

        self.assertEqual(stopped["state"], "error")
        self.assertEqual(ramdisk._load_manifest(required=True)["state"], "error")

    def test_managed_readiness_requires_verified_health_response(self):
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def read(self):
                return b'{"status":"ok"}'

        record = {"pid": 123, "port": 8123, "log": "/tmp/engine.log"}
        with mock.patch.object(
            ramdisk, "_process_matches", return_value=(True, "running", {})
        ):
            ramdisk._wait_managed_ready(
                record,
                timeout=1,
                api_key="secret",
                urlopen=mock.Mock(return_value=Response()),
            )
        self.assertIn("ready_at", record)

    @requires_linux_operational
    def test_destroy_refuses_replaced_mount_identity(self):
        mount_path = "/mnt/colibri-test"
        manifest = self.manifest(mount_paths=[mount_path])
        ramdisk._atomic_json(ramdisk._manifest_path(), manifest)
        replacement = {
            "mount_id": 5,
            "device": "0:10",
            "filesystem": "tmpfs",
            "source": "tmpfs",
        }
        args = argparse.Namespace(yes=True)
        with mock.patch.object(ramdisk, "_mount_at", return_value=replacement), mock.patch.object(
            ramdisk, "_umount_path"
        ) as unmount:
            with self.assertRaisesRegex(ramdisk.RamdiskError, "foreign or replaced"):
                ramdisk.destroy(args)
        unmount.assert_not_called()
        persisted = ramdisk._read_json(ramdisk._manifest_path())
        self.assertEqual(persisted["state"], "error")
        self.assertIn(mount_path, persisted["recovery"]["retained_mounts"])

    @requires_linux_operational
    def test_destroy_retains_manifest_for_unrecorded_surviving_mount(self):
        mount_path = "/mnt/colibri-test"
        manifest = self.manifest(state="error", mount_paths=[mount_path])
        manifest["mounts"] = []
        ramdisk._atomic_json(ramdisk._manifest_path(), manifest)
        surviving = {
            "mount_id": 17,
            "device": "0:77",
            "filesystem": "tmpfs",
            "source": "tmpfs",
        }
        with mock.patch.object(ramdisk, "_mount_table", return_value=[]), mock.patch.object(
            ramdisk, "_mount_at", return_value=surviving
        ), mock.patch.object(ramdisk, "_umount_path") as unmount:
            with self.assertRaisesRegex(ramdisk.RamdiskError, "unverified surviving mount"):
                ramdisk.destroy(argparse.Namespace(yes=True))
        unmount.assert_not_called()
        self.assertTrue(os.path.exists(ramdisk._manifest_path()))

    def test_destroy_retains_absent_pending_mount_that_helper_may_publish_later(self):
        mount_path = "/mnt/colibri-test"
        manifest = self.manifest(state="error", mount_paths=[mount_path])
        manifest["mounts"][0].pop("identity")
        manifest["mounts"][0]["ownership"] = "pending"
        manifest["error"] = "mount helper outcome is unknown"
        ramdisk._atomic_json(ramdisk._manifest_path(), manifest)

        with mock.patch.object(
            ramdisk, "_mount_table", return_value=[]
        ) as mount_table, mock.patch.object(
            # Even an absent-now observation cannot clear an in-flight helper.
            ramdisk, "_mount_at", return_value=None
        ) as mount_at, mock.patch.object(
            ramdisk, "_umount_path"
        ) as unmount, mock.patch.object(
            ramdisk, "_durable_unlink"
        ) as unlink:
            with self.assertRaisesRegex(
                ramdisk.RamdiskError,
                "mount helper outcome is unknown.*pending",
            ):
                ramdisk.destroy(argparse.Namespace(yes=True))

        mount_table.assert_not_called()
        mount_at.assert_not_called()
        unmount.assert_not_called()
        unlink.assert_not_called()
        persisted = ramdisk._load_manifest(required=True)
        self.assertEqual(persisted["state"], "error")
        self.assertEqual(persisted["mounts"][0]["ownership"], "pending")
        self.assertEqual(
            persisted["recovery"]["retained_mounts"],
            [mount_path],
        )

    @requires_linux_operational
    def test_destroy_preflights_every_busy_mount_before_unmounting(self):
        paths = ["/mnt/colibri-test/node0", "/mnt/colibri-test/node1"]
        manifest = self.manifest(mount_paths=paths)
        ramdisk._atomic_json(ramdisk._manifest_path(), manifest)

        def mounted(path):
            record = next(item for item in manifest["mounts"] if item["path"] == path)
            return dict(record["identity"], filesystem="tmpfs", source="tmpfs")

        with mock.patch.object(ramdisk, "_mount_at", side_effect=mounted), mock.patch.object(
            ramdisk, "_validate_mount", side_effect=lambda record, plan: mounted(record["path"])
        ), mock.patch.object(ramdisk, "_validate_namespace"), mock.patch.object(
            ramdisk, "_busy_mount_references", side_effect=[[], [999]]
        ), mock.patch.object(ramdisk, "_umount_path") as unmount:
            with self.assertRaisesRegex(ramdisk.RamdiskError, "busy"):
                ramdisk.destroy(argparse.Namespace(yes=True))
        unmount.assert_not_called()

    @requires_linux_operational
    def test_destroy_rejects_nested_child_mounts_before_any_unmount(self):
        paths = ["/mnt/colibri-test/node0", "/mnt/colibri-test/node1"]
        manifest = self.manifest(mount_paths=paths)
        ramdisk._atomic_json(ramdisk._manifest_path(), manifest)
        child = {
            "mount_id": 99,
            "path": paths[1] + "/foreign-child",
            "filesystem": "ext4",
            "source": "/dev/loop0",
        }
        with mock.patch.object(ramdisk, "_mount_table", return_value=[child]), mock.patch.object(
            ramdisk, "_umount_path"
        ) as unmount:
            with self.assertRaisesRegex(ramdisk.RamdiskError, "nested child mounts"):
                ramdisk.destroy(argparse.Namespace(yes=True))
        unmount.assert_not_called()
        persisted = ramdisk._read_json(ramdisk._manifest_path())
        self.assertEqual(persisted["state"], "error")
        self.assertEqual(
            persisted["recovery"]["retained_mounts"],
            paths,
        )

    @requires_linux_operational
    def test_destroy_persists_recovery_state_when_kernel_unmount_fails(self):
        mount_path = "/mnt/colibri-test"
        manifest = self.manifest(state="stopped", mount_paths=[mount_path])
        ramdisk._atomic_json(ramdisk._manifest_path(), manifest)
        actual = dict(
            manifest["mounts"][0]["identity"],
            filesystem="tmpfs",
            source="tmpfs",
        )

        with mock.patch.object(
            ramdisk, "_mount_table", return_value=[]
        ), mock.patch.object(
            ramdisk, "_mount_at", return_value=actual
        ), mock.patch.object(
            ramdisk, "_validate_mount", return_value=actual
        ), mock.patch.object(
            ramdisk, "_validate_namespace"
        ), mock.patch.object(
            ramdisk, "_busy_mount_references", return_value=[]
        ), mock.patch.object(
            ramdisk,
            "_umount_path",
            side_effect=ramdisk.RamdiskError("kernel refused unmount"),
        ):
            with self.assertRaisesRegex(
                ramdisk.RamdiskError,
                "kernel refused unmount",
            ):
                ramdisk.destroy(argparse.Namespace(yes=True))

        persisted = ramdisk._read_json(ramdisk._manifest_path())
        self.assertEqual(persisted["state"], "error")
        self.assertIn(
            "kernel refused unmount",
            persisted["destroy_error"],
        )
        self.assertEqual(
            persisted["recovery"]["retained_mounts"],
            [mount_path],
        )

    def test_destroy_revalidates_mount_identity_immediately_before_unmount(self):
        mount_path = "/mnt/colibri-test"
        manifest = self.manifest(state="stopped", mount_paths=[mount_path])
        ramdisk._atomic_json(ramdisk._manifest_path(), manifest)
        actual = dict(
            manifest["mounts"][0]["identity"],
            filesystem="tmpfs",
            source="tmpfs",
        )
        replacement = dict(actual, mount_id=81, device="0:81")

        with mock.patch.object(
            ramdisk, "_mount_table", return_value=[]
        ), mock.patch.object(
            ramdisk, "_mount_at", side_effect=[actual, replacement]
        ), mock.patch.object(
            ramdisk, "_validate_mount", return_value=actual
        ), mock.patch.object(
            ramdisk, "_validate_namespace"
        ), mock.patch.object(
            ramdisk, "_busy_mount_references", return_value=[]
        ), mock.patch.object(
            ramdisk, "_umount_path"
        ) as unmount:
            with self.assertRaisesRegex(
                ramdisk.RamdiskError,
                "foreign or replaced mount",
            ):
                ramdisk.destroy(argparse.Namespace(yes=True))

        unmount.assert_not_called()
        persisted = ramdisk._load_manifest(required=True)
        self.assertEqual(persisted["state"], "error")
        self.assertEqual(
            persisted["recovery"]["retained_mounts"],
            [mount_path],
        )

    def test_destroy_requires_post_unmount_absence(self):
        mount_path = "/mnt/colibri-test"
        for replacement_id in (None, 91):
            with self.subTest(replacement_id=replacement_id):
                manifest = self.manifest(
                    state="stopped",
                    mount_paths=[mount_path],
                )
                ramdisk._atomic_json(ramdisk._manifest_path(), manifest)
                actual = dict(
                    manifest["mounts"][0]["identity"],
                    filesystem="tmpfs",
                    source="tmpfs",
                )
                after = (
                    actual
                    if replacement_id is None
                    else dict(
                        actual,
                        mount_id=replacement_id,
                        device="0:%d" % replacement_id,
                    )
                )

                with mock.patch.object(
                    ramdisk, "_mount_table", return_value=[]
                ), mock.patch.object(
                    ramdisk,
                    "_mount_at",
                    side_effect=[actual, actual, actual, after],
                ), mock.patch.object(
                    ramdisk, "_validate_mount", return_value=actual
                ), mock.patch.object(
                    ramdisk, "_validate_namespace"
                ), mock.patch.object(
                    ramdisk, "_busy_mount_references", return_value=[]
                ), mock.patch.object(
                    ramdisk, "_umount_path"
                ) as unmount, mock.patch.object(
                    ramdisk, "_durable_unlink"
                ) as unlink:
                    with self.assertRaisesRegex(
                        ramdisk.RamdiskError,
                        "remains or was replaced",
                    ):
                        ramdisk.destroy(argparse.Namespace(yes=True))

                unmount.assert_called_once()
                unlink.assert_not_called()
                persisted = ramdisk._load_manifest(required=True)
                self.assertEqual(persisted["state"], "error")
                self.assertEqual(
                    persisted["recovery"]["retained_mounts"],
                    [mount_path],
                )

    def test_destroy_rechecks_identity_after_busy_scan(self):
        mount_path = "/mnt/colibri-test"
        manifest = self.manifest(state="stopped", mount_paths=[mount_path])
        ramdisk._atomic_json(ramdisk._manifest_path(), manifest)
        actual = dict(
            manifest["mounts"][0]["identity"],
            filesystem="tmpfs",
            source="tmpfs",
        )
        replacement = dict(actual, mount_id=92, device="0:92")
        current = {"identity": actual}
        busy_calls = {"count": 0}

        def busy(_path, hardware=None):
            busy_calls["count"] += 1
            if busy_calls["count"] == 2:
                current["identity"] = replacement
            return []

        with mock.patch.object(
            ramdisk, "_mount_table", return_value=[]
        ), mock.patch.object(
            ramdisk,
            "_mount_at",
            side_effect=lambda ignored: current["identity"],
        ), mock.patch.object(
            ramdisk,
            "_validate_mount",
            side_effect=lambda ignored, plan: current["identity"],
        ), mock.patch.object(
            ramdisk, "_validate_namespace"
        ), mock.patch.object(
            ramdisk, "_busy_mount_references", side_effect=busy
        ), mock.patch.object(
            ramdisk, "_umount_path"
        ) as unmount:
            with self.assertRaisesRegex(
                ramdisk.RamdiskError,
                "after busy scan",
            ):
                ramdisk.destroy(argparse.Namespace(yes=True))

        unmount.assert_not_called()
        persisted = ramdisk._load_manifest(required=True)
        self.assertEqual(persisted["state"], "error")
        self.assertEqual(
            persisted["recovery"]["retained_mounts"],
            [mount_path],
        )

    @requires_linux_operational
    def test_busy_mount_scan_includes_the_manager_process(self):
        held = os.path.join(self.root, "held-mount")
        child = os.path.join(held, "inside")
        os.makedirs(child)
        previous = os.getcwd()
        try:
            os.chdir(child)
            # Exercise the root-only procfs implementation without enumerating
            # unrelated host processes. The unprivileged fuser command and
            # parser contracts are covered independently in platform tests.
            with mock.patch.object(
                linux_ops.os,
                "listdir",
                return_value=[str(os.getpid())],
            ):
                self.assertIn(
                    os.getpid(),
                    linux_ops._busy_mount_references_proc(held),
                )
        finally:
            os.chdir(previous)

    @requires_linux_operational
    def test_dashboard_rss_sums_verified_wrapper_and_engine_group(self):
        record = {
            "pid": 101,
            "pgid": 101,
            "uid": host_uid(),
            "nonce": "a" * 48,
        }
        members = [
            {"pid": 101, "uid": host_uid(), "nonce": "a" * 48},
            {"pid": 102, "uid": host_uid(), "nonce": "a" * 48},
        ]

        def proc_text(path, default=""):
            if path == "/proc/101/status":
                return "VmRSS:\t100 kB\n"
            if path == "/proc/102/status":
                return "VmRSS:\t900 kB\n"
            return default

        with mock.patch.object(
            ramdisk, "_process_matches", return_value=(True, "running", {})
        ), mock.patch.object(
            ramdisk, "_process_group_members", return_value=(members, [])
        ), mock.patch.object(ramdisk, "_read_text", side_effect=proc_text):
            metrics = ramdisk._managed_process_metrics(record)
        self.assertEqual(metrics["rss_bytes"], 1000 * 1024)
        self.assertEqual(metrics["rss_processes"], 2)

    def test_status_absent_is_versioned(self):
        report = ramdisk.status()
        self.assertEqual(report["schema"], ramdisk.STATUS_SCHEMA)
        self.assertEqual(report["state"], "absent")

    def test_status_exposes_sanitized_actionable_recovery(self):
        manifest = self.manifest(state="error")
        secret_nonce = "9" * 48
        secret_merge_id = "8" * 32
        manifest["launch_error"] = "engine readiness failed"
        manifest["cleanup_errors"] = ["group absence unproven"]
        manifest["recovery"] = {
            "operation": "start",
            "state": "attention-required",
            "retained_mounts": [
                "/mnt/colibri-test",
                {"private_nonce": secret_nonce},
            ],
            "retained_processes": [
                {
                    "pid": 14001,
                    "pgid": 14001,
                    "node": None,
                    "state_dir": os.path.join(self.root, "retained-state"),
                    "usage_baseline": {"0:1": 19},
                    "usage_merge_id": "7" * 32,
                    "error": "process group still live",
                }
            ],
        }
        manifest["pending_launches"] = [
            {
                "operation_id": "start:" + secret_merge_id,
                "nonce": secret_nonce,
                "port": 8000,
                "node": None,
                "state_dir": os.path.join(self.root, "pending-state"),
                "usage_baseline": {"0:1": 23},
                "usage_merge_id": secret_merge_id,
            }
        ]
        with mock.patch.object(
            ramdisk, "_load_manifest", return_value=manifest
        ), mock.patch.object(
            ramdisk, "_mount_at", return_value=None
        ):
            report = ramdisk.status(deep=False)

        self.assertEqual(report["recovery"]["operation"], "start")
        self.assertEqual(
            report["recovery"]["retained_processes"][0]["pid"],
            14001,
        )
        self.assertEqual(
            report["recovery"]["pending_launches"][0]["port"],
            8000,
        )
        self.assertIn(
            "engine readiness failed",
            report["recovery"]["errors"]["launch_error"],
        )
        serialized = json.dumps(report, sort_keys=True)
        self.assertNotIn(secret_nonce, serialized)
        self.assertNotIn(secret_merge_id, serialized)
        self.assertNotIn("usage_baseline", serialized)
        self.assertNotIn("usage_merge_id", serialized)

    def test_deep_status_preserves_recovery_when_source_scan_raises_oserror(self):
        manifest = self.manifest(state="error")
        manifest["recovery"] = {
            "operation": "destroy",
            "state": "attention-required",
            "retained_mounts": [manifest["mounts"][0]["path"]],
            "released_mounts": [],
        }
        with mock.patch.object(
            ramdisk, "_load_manifest", return_value=manifest
        ), mock.patch.object(
            ramdisk,
            "_source_still_matches",
            side_effect=OSError("source shard became unreadable"),
        ), mock.patch.object(
            ramdisk, "_mount_at", return_value=None
        ):
            report = ramdisk.status(deep=True)

        self.assertFalse(report["source_fingerprint_verified"])
        self.assertIn(
            "source shard became unreadable",
            report["source_fingerprint_error"],
        )
        self.assertEqual(
            report["recovery"]["retained_mounts"],
            [manifest["mounts"][0]["path"]],
        )
        self.assertIn("`coli ramdisk destroy`", report["recovery"]["action"])

    def test_status_propagates_control_flow_from_probe_seams(self):
        mount_interrupt = KeyboardInterrupt("mount probe interrupted")
        manifest = self.manifest(state="ready")
        with mock.patch.object(
            ramdisk, "_load_manifest", return_value=manifest
        ), mock.patch.object(
            ramdisk, "_mount_at", side_effect=mount_interrupt
        ):
            with self.assertRaises(KeyboardInterrupt):
                ramdisk.status(deep=False)

        process_manifest = self.manifest(
            state="running",
            processes=[{"pid": 14003}],
        )
        identity_interrupt = ramdisk._TuiTerminationSignal(signal.SIGTERM)
        with mock.patch.object(
            ramdisk, "_load_manifest", return_value=process_manifest
        ), mock.patch.object(
            ramdisk, "_mount_at", return_value=None
        ), mock.patch.object(
            ramdisk, "_process_matches", side_effect=identity_interrupt
        ):
            with self.assertRaises(ramdisk._TuiTerminationSignal):
                ramdisk.status(deep=False)

        liveness_interrupt = ramdisk._TuiTerminationSignal(signal.SIGINT)
        with mock.patch.object(
            ramdisk, "_load_manifest", return_value=process_manifest
        ), mock.patch.object(
            ramdisk, "_mount_at", return_value=None
        ), mock.patch.object(
            ramdisk,
            "_process_matches",
            return_value=(False, "not-running", None),
        ), mock.patch.object(
            ramdisk,
            "_managed_child_liveness",
            side_effect=liveness_interrupt,
        ):
            with self.assertRaises(ramdisk._TuiTerminationSignal):
                ramdisk.status(deep=False)

    def test_status_gives_conservative_mount_only_recovery_action(self):
        manifest = self.manifest(state="error")
        manifest["recovery"] = {
            "operation": "destroy",
            "state": "attention-required",
            "retained_mounts": ["/mnt/colibri-test"],
            "released_mounts": [],
        }
        with mock.patch.object(
            ramdisk, "_load_manifest", return_value=manifest
        ), mock.patch.object(
            ramdisk, "_mount_at", return_value=None
        ):
            report = ramdisk.status(deep=False)

        action = report["recovery"]["action"]
        self.assertIn("mount identity", action)
        self.assertIn("nested mounts", action)
        self.assertIn("busy references", action)
        self.assertIn("`coli ramdisk destroy`", action)
        self.assertIn("only after confirming it is safe", action)
        self.assertNotIn("--yes", action)

    def test_status_synthesizes_recovery_for_hard_crash_pending_mount(self):
        manifest = self.manifest(state="preparing")
        manifest["mounts"][0].pop("identity")
        manifest["mounts"][0]["ownership"] = "pending"
        mount_path = manifest["mounts"][0]["path"]

        with mock.patch.object(
            ramdisk, "_load_manifest", return_value=manifest
        ), mock.patch.object(
            ramdisk, "_mount_at", return_value=None
        ):
            report = ramdisk.status(deep=False)

        recovery = report["recovery"]
        self.assertEqual(recovery["operation"], "prepare")
        self.assertEqual(recovery["state"], "attention-required")
        self.assertEqual(recovery["retained_mounts"], [mount_path])
        self.assertIn("pending ownership", recovery["action"])
        self.assertIn("`coli ramdisk destroy`", recovery["action"])

    def test_stopped_process_group_is_revalidated_by_start_stop_and_status(self):
        manifest = self.manifest(
            state="stopped",
            processes=[
                {
                    "pid": 14002,
                    "stopped_at": "2026-08-01T00:00:00Z",
                    "usage_merged_at": "2026-08-01T00:00:00Z",
                }
            ],
        )
        ramdisk._atomic_json(ramdisk._manifest_path(), manifest)
        running = (True, "running-group", {"pgid": 14002})

        with mock.patch.object(
            ramdisk, "_process_matches", return_value=running
        ), mock.patch.object(
            ramdisk, "_managed_child_liveness", return_value=None
        ), mock.patch.object(
            ramdisk, "_assert_effective_masks_unchanged"
        ), mock.patch.object(
            ramdisk, "_assert_ready_mounts"
        ), mock.patch.object(
            ramdisk.subprocess, "Popen"
        ) as popen:
            with self.assertRaisesRegex(
                ramdisk.RamdiskError,
                "stopped-record-process-group-live",
            ):
                ramdisk.start.__wrapped__(
                    argparse.Namespace(base_port=None),
                    cli_path=sys.executable,
                )
            with self.assertRaisesRegex(
                ramdisk.RamdiskError,
                "stopped-record-process-group-live",
            ):
                ramdisk.stop()
            report = ramdisk.status(deep=False)

        popen.assert_not_called()
        self.assertTrue(report["processes"][0]["running"])
        self.assertTrue(report["processes"][0]["attention_required"])
        self.assertEqual(
            report["processes"][0]["reason"],
            "stopped-record-process-group-live",
        )

    @requires_linux_operational
    def test_manifest_rejects_volatile_durable_state(self):
        manifest = self.manifest()
        ramdisk._atomic_json(ramdisk._manifest_path(), manifest)
        with mock.patch.object(ramdisk, "_filesystem_for_path", return_value="tmpfs"):
            with self.assertRaisesRegex(ramdisk.RamdiskError, "volatile"):
                ramdisk._load_manifest(required=True)
