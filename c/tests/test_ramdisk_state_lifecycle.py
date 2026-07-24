"""RAM-disk durable state, safety, and lifecycle tests."""

if __package__:
    from .ramdisk_test_support import *  # noqa: F401,F403
else:
    from ramdisk_test_support import *  # noqa: F401,F403


class StateAndSafetyTest(unittest.TestCase):
    FINGERPRINT = "sha256:" + ("a" * 64)

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

    def test_manifest_rejects_missing_nonce_before_process_signaling(self):
        manifest = self.manifest(
            state="running", processes=[{"pid": 12345}]
        )
        manifest["processes"][0].pop("nonce")
        ramdisk._atomic_json(ramdisk._manifest_path(), manifest)
        with self.assertRaisesRegex(ramdisk.RamdiskError, "unsafe managed process"):
            ramdisk._load_manifest(required=True)

    def test_manifest_rejects_mount_layout_outside_v1_root(self):
        manifest = self.manifest()
        manifest["plan"]["mount_root"] = os.path.join(self.root, "mount")
        manifest["plan"]["mounts"][0]["path"] = manifest["plan"]["mount_root"]
        manifest["mounts"][0]["path"] = manifest["plan"]["mount_root"]
        ramdisk._atomic_json(ramdisk._manifest_path(), manifest)
        with self.assertRaisesRegex(ramdisk.RamdiskError, "unsafe mount root"):
            ramdisk._load_manifest(required=True)

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
        ), mock.patch.object(ramdisk.urllib.request, "urlopen", return_value=Response()):
            ramdisk._wait_managed_ready(record, timeout=1, api_key="secret")
        self.assertIn("ready_at", record)

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

    def test_busy_mount_scan_includes_the_manager_process(self):
        held = os.path.join(self.root, "held-mount")
        child = os.path.join(held, "inside")
        os.makedirs(child)
        previous = os.getcwd()
        try:
            os.chdir(child)
            self.assertIn(os.getpid(), ramdisk._busy_mount_references(held))
        finally:
            os.chdir(previous)

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

    def test_manifest_rejects_volatile_durable_state(self):
        manifest = self.manifest()
        ramdisk._atomic_json(ramdisk._manifest_path(), manifest)
        with mock.patch.object(ramdisk, "_filesystem_for_path", return_value="tmpfs"):
            with self.assertRaisesRegex(ramdisk.RamdiskError, "volatile"):
                ramdisk._load_manifest(required=True)
