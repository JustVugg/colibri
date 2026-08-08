"""Headless presentation, review, cancellation, and privilege contracts."""

if __package__:
    from .ramdisk_test_support import *  # noqa: F401,F403
else:
    from ramdisk_test_support import *  # noqa: F401,F403

from ramdisk_support.presentation import _accelerator_review



class HeadlessPresentationContractTest(unittest.TestCase):
    def setUp(self):
        for name, value in (
            ("_ensure_busy_mount_scan_available", None),
            ("_busy_mount_references", []),
        ):
            patcher = mock.patch.object(ramdisk, name, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)


    def test_human_status_renders_only_sanitized_recovery_fields(self):
        secret_nonce = "a" * 48
        secret_merge_id = "b" * 32
        report = {
            "present": True,
            "state": "error",
            "mounts": [],
            "processes": [],
            "recovery": {
                "operation": "start",
                "state": "attention-required",
                "retained_mounts": ["/mnt/colibri-test"],
                "released_mounts": [],
                "retained_processes": [
                    {
                        "pid": 15001,
                        "state_dir": "/state/retained",
                        "error": "absence unproven",
                    }
                ],
                "pending_launches": [
                    {
                        "node": None,
                        "port": 8000,
                        "state_dir": "/state/pending",
                        # Unknown/private keys must never be rendered.
                        "nonce": secret_nonce,
                        "usage_merge_id": secret_merge_id,
                    }
                ],
                "errors": {"launch_error": "readiness failed"},
                "action": "Inspect and reconcile before retrying.",
            },
        }
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            ramdisk._human_status(report)

        rendered = output.getvalue()
        self.assertIn("recovery: start / attention-required", rendered)
        self.assertIn("retained mount: /mnt/colibri-test", rendered)
        self.assertIn("retained PID 15001", rendered)
        self.assertIn("outcome-unknown launch node None port 8000", rendered)
        self.assertIn("readiness failed", rendered)
        self.assertIn("action: Inspect and reconcile", rendered)
        self.assertNotIn(secret_nonce, rendered)
        self.assertNotIn(secret_merge_id, rendered)
        self.assertNotIn("usage_merge_id", rendered)

    def test_four_node_default_is_one_shared_model_and_one_engine(self):
        with ModelFixture() as fixture:
            plan = ramdisk.build_plan(
                plan_args(fixture.root), hardware=hardware_fixture(nodes=4)
            )

        placement = ramdisk._placement_summary(plan, base_port=8000)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            ramdisk._human_plan(plan)
        rendered = output.getvalue()

        self.assertEqual(plan["topology"], "interleaved")
        self.assertEqual(plan["staging"]["replica_count"], 1)
        self.assertEqual(placement["copy_count"], 1)
        self.assertEqual(placement["engine_count"], 1)
        self.assertIn("Single shared model", placement["title"])
        self.assertIn("1 complete model copy", placement["cost"])
        self.assertIn("1 engine", placement["cost"])
        self.assertIn("spread across 4 NUMA nodes", placement["explanation"])
        self.assertIn("1 complete model copy", rendered)
        self.assertIn("endpoints after start: port 8000", rendered)

    def test_gpu_review_names_exact_devices_layout_and_projection(self):
        with ModelFixture() as fixture:
            hardware = hardware_fixture(nodes=2)
            hardware["gpus"] = [
                {
                    "index": index,
                    "name": "GPU %d" % index,
                    "uuid": "GPU-test-%d" % index,
                    "pci_bus_id": bus,
                    "numa_node": index,
                    "locality": "resolved",
                    "total_bytes": 32 * ramdisk.GIB,
                    "free_bytes": 28 * ramdisk.GIB,
                }
                for index, bus in (
                    (0, "0000:41:00.0"),
                    (1, "0000:c1:00.0"),
                )
            ]
            hardware["gpu_discovery"] = {
                "status": "available",
                "error": None,
            }
            plan = ramdisk.build_plan(
                plan_args(
                    fixture.root,
                    gpu="0,1",
                    gpu_layout="dense-attention-sharded",
                ),
                hardware=hardware,
            )

        review = _accelerator_review(plan)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            ramdisk._human_plan(plan)
        rendered = output.getvalue()

        self.assertEqual(review["indices"], "0,1")
        self.assertEqual(review["layout"], "dense-attention-sharded")
        self.assertIsNotNone(review["dense_gpu_gib"])
        self.assertIsNotNone(review["expert_headroom_gib"])
        self.assertIn("0,1", rendered)
        self.assertIn("dense-attention-sharded", rendered)
        self.assertIn("GPU projection", rendered)
        self.assertIn("expert headroom", rendered)

    def test_four_node_replica_mode_names_every_full_copy_and_endpoint(self):
        with ModelFixture() as fixture:
            shared = ramdisk.build_plan(
                plan_args(fixture.root), hardware=hardware_fixture(nodes=4)
            )
            replicated = ramdisk.build_plan(
                plan_args(fixture.root, topology="per-node"),
                hardware=hardware_fixture(nodes=4),
            )

        placement = ramdisk._placement_summary(replicated, base_port=8000)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            ramdisk._human_plan(replicated)
        rendered = output.getvalue()

        self.assertEqual(placement["copy_count"], 4)
        self.assertEqual(placement["engine_count"], 4)
        self.assertIn("Independent full-model replicas", placement["title"])
        self.assertIn("4 complete model copies", placement["cost"])
        self.assertIn("4 independent engines", placement["cost"])
        self.assertIn("ports 8000, 8001, 8002, 8003", placement["endpoints"])
        self.assertIn("replication, not model sharding", placement["explanation"])
        self.assertIn("4 complete model copies", rendered)
        self.assertIn("4 independent engines", rendered)
        self.assertIn("not model sharding", rendered)
        self.assertNotEqual(
            ramdisk._plan_confirmation_token(shared),
            ramdisk._plan_confirmation_token(replicated),
        )

    def test_prepare_rejects_a_plan_changed_after_review(self):
        with ModelFixture() as fixture:
            shared = ramdisk.build_plan(
                plan_args(fixture.root), hardware=hardware_fixture(nodes=2)
            )
            replicated = ramdisk.build_plan(
                plan_args(fixture.root, topology="per-node"),
                hardware=hardware_fixture(nodes=2),
            )
            reviewed = ramdisk._plan_confirmation_token(shared)
            with mock.patch.object(ramdisk, "_load_manifest", return_value=None), mock.patch.object(
                ramdisk, "build_plan", return_value=replicated
            ), mock.patch.object(ramdisk, "_mount_tmpfs") as mount:
                with self.assertRaisesRegex(ramdisk.RamdiskError, "changed since review"):
                    ramdisk.prepare.__wrapped__(
                        plan_args(fixture.root, yes=True),
                        display_plan=False,
                        expected_plan_token=reviewed,
                    )
        mount.assert_not_called()

    def test_destroy_rejects_a_replacement_after_review(self):
        with ModelFixture() as fixture:
            plan = ramdisk.build_plan(
                plan_args(fixture.root), hardware=hardware_fixture(nodes=2)
            )
        mount = dict(plan["mounts"][0])
        mount["identity"] = {"mount_id": 41, "device": "0:41"}
        reviewed_manifest = {
            "version": ramdisk.MANIFEST_VERSION,
            "deployment_id": "a" * 32,
            "created_at": "2026-07-22T10:00:00+00:00",
            "model_fingerprint": plan["model"]["fingerprint"],
            "plan": plan,
            "mounts": [mount],
            "processes": [],
        }
        replacement_manifest = dict(reviewed_manifest)
        replacement_mount = dict(mount)
        replacement_mount["identity"] = {"mount_id": 42, "device": "0:42"}
        replacement_manifest.update(
            {
                "deployment_id": "b" * 32,
                "created_at": "2026-07-22T10:00:01+00:00",
                "mounts": [replacement_mount],
            }
        )
        reviewed_token = ramdisk._manifest_confirmation_token(reviewed_manifest)

        with mock.patch.object(
            ramdisk, "_load_manifest", return_value=replacement_manifest
        ), mock.patch.object(ramdisk, "_confirm") as confirm, mock.patch.object(
            ramdisk, "_umount_path"
        ) as unmount, mock.patch.object(ramdisk, "_destroy_locked") as destroy_locked:
            with self.assertRaisesRegex(ramdisk.RamdiskError, "changed since review"):
                ramdisk.destroy(
                    argparse.Namespace(yes=True),
                    expected_manifest_token=reviewed_token,
                )

        confirm.assert_not_called()
        unmount.assert_not_called()
        destroy_locked.assert_not_called()

    def test_destroy_confirmation_expires_when_process_state_changes(self):
        with ModelFixture() as fixture:
            plan = ramdisk.build_plan(
                plan_args(fixture.root), hardware=hardware_fixture(nodes=2)
            )
        manifest = {
            "version": ramdisk.MANIFEST_VERSION,
            "deployment_id": "a" * 32,
            "created_at": "2026-07-22T10:00:00+00:00",
            "state": "ready",
            "model_fingerprint": plan["model"]["fingerprint"],
            "plan": plan,
            "mounts": [],
            "processes": [],
        }
        reviewed_token = ramdisk._manifest_confirmation_token(manifest)
        started = dict(manifest)
        started["state"] = "starting"
        started["processes"] = [
            {
                "pid": 1234,
                "pgid": 1234,
                "uid": 1000,
                "starttime": 5678,
                "nonce": "managed",
                "port": 8000,
                "node": None,
            }
        ]

        self.assertNotEqual(
            reviewed_token,
            ramdisk._manifest_confirmation_token(started),
        )

    def test_destroy_confirmation_expires_when_endpoint_changes(self):
        with ModelFixture() as fixture:
            plan = ramdisk.build_plan(
                plan_args(fixture.root), hardware=hardware_fixture(nodes=2)
            )
        manifest = {
            "version": ramdisk.MANIFEST_VERSION,
            "deployment_id": "a" * 32,
            "created_at": "2026-07-22T10:00:00+00:00",
            "state": "ready",
            "base_port": 8000,
            "model_fingerprint": plan["model"]["fingerprint"],
            "plan": plan,
            "mounts": [],
            "processes": [],
        }
        reviewed_token = ramdisk._manifest_confirmation_token(manifest)
        changed_endpoint = dict(manifest, base_port=9000)

        self.assertNotEqual(
            reviewed_token,
            ramdisk._manifest_confirmation_token(changed_endpoint),
        )


    def test_legacy_stopped_manifest_recovers_its_previous_base_port(self):
        manifest = {
            "processes": [
                {
                    "port": 9100,
                    "node": None,
                    "stopped_at": "2026-07-22T10:00:00+00:00",
                }
            ],
            "ports": [9100],
            "mounts": [{"node": None}],
        }
        self.assertEqual(ramdisk._persisted_base_port(manifest), 9100)

    def test_manifest_rejects_boolean_base_port(self):
        with mock.patch.object(
            ramdisk,
            "_read_json",
            return_value={"version": ramdisk.MANIFEST_VERSION, "base_port": True},
        ):
            with self.assertRaisesRegex(ramdisk.RamdiskError, "base port"):
                ramdisk._load_manifest(required=True)

    def test_cancelled_prepare_stops_before_mounting(self):
        cancel = threading.Event()
        cancel.set()
        with ModelFixture() as fixture:
            plan = ramdisk.build_plan(
                plan_args(fixture.root), hardware=hardware_fixture(nodes=2)
            )
            with mock.patch.object(
                ramdisk, "_load_manifest", return_value=None
            ), mock.patch.object(ramdisk, "build_plan", return_value=plan), mock.patch.object(
                ramdisk, "_mount_tmpfs"
            ) as mount:
                with self.assertRaisesRegex(
                    ramdisk._OperationCancelled, "cancelled by user"
                ):
                    ramdisk.prepare.__wrapped__(
                        plan_args(fixture.root, yes=True),
                        display_plan=False,
                        cancel_event=cancel,
                    )
        mount.assert_not_called()

    @requires_linux_operational
    def test_cancelled_prepare_does_not_hide_rollback_failure(self):
        cancel = threading.Event()
        with ModelFixture() as fixture:
            plan = ramdisk.build_plan(
                plan_args(fixture.root), hardware=hardware_fixture(nodes=2)
            )
        actual = {
            "mount_id": 91,
            "device": "0:91",
            "filesystem": "tmpfs",
            "source": "tmpfs",
        }

        def cancel_copy(*args, **kwargs):
            cancel.set()
            ramdisk._raise_if_cancelled(cancel)

        with mock.patch.object(
            ramdisk, "_load_manifest", return_value=None
        ), mock.patch.object(ramdisk, "build_plan", return_value=plan), mock.patch.object(
            ramdisk, "_save_manifest"
        ), mock.patch.object(
            ramdisk,
            "_mount_at",
            side_effect=[None, actual, actual, actual, actual, actual],
        ), mock.patch.object(ramdisk, "_mount_tmpfs"), mock.patch.object(
            ramdisk, "_validate_mount", return_value=actual
        ), mock.patch.object(ramdisk, "_populate_mount", side_effect=cancel_copy), mock.patch.object(
            ramdisk, "_umount_path", side_effect=OSError("sudo ticket expired")
        ):
            with self.assertRaises(ramdisk.RamdiskError) as raised:
                ramdisk.prepare.__wrapped__(
                    plan_args(fixture.root, yes=True),
                    display_plan=False,
                    cancel_event=cancel,
                )

        self.assertNotIsInstance(raised.exception, ramdisk._OperationCancelled)
        self.assertIn("rollback/reporting errors", str(raised.exception))
        self.assertIn("sudo ticket expired", str(raised.exception))

    @requires_linux_operational
    def test_clean_prepare_cancellation_removes_recovery_manifest(self):
        cancel = threading.Event()
        with ModelFixture() as fixture:
            plan = ramdisk.build_plan(
                plan_args(fixture.root), hardware=hardware_fixture(nodes=2)
            )
        actual = {
            "mount_id": 92,
            "device": "0:92",
            "filesystem": "tmpfs",
            "source": "tmpfs",
        }

        def cancel_copy(*args, **kwargs):
            cancel.set()
            ramdisk._raise_if_cancelled(cancel)

        with mock.patch.object(
            ramdisk, "_load_manifest", return_value=None
        ), mock.patch.object(ramdisk, "build_plan", return_value=plan), mock.patch.object(
            ramdisk, "_save_manifest"
        ), mock.patch.object(
            ramdisk,
            "_mount_at",
            side_effect=[
                None,
                actual,
                actual,
                actual,
                actual,
                actual,
                None,
            ],
        ), mock.patch.object(ramdisk, "_mount_tmpfs"), mock.patch.object(
            ramdisk, "_validate_mount", return_value=actual
        ), mock.patch.object(ramdisk, "_populate_mount", side_effect=cancel_copy), mock.patch.object(
            ramdisk, "_umount_path"
        ) as unmount, mock.patch.object(ramdisk, "_durable_unlink") as unlink:
            with self.assertRaises(ramdisk._OperationCancelled):
                ramdisk.prepare.__wrapped__(
                    plan_args(fixture.root, yes=True),
                    display_plan=False,
                    cancel_event=cancel,
                )

        unmount.assert_called_once_with(plan["mounts"][0]["path"], plan["hardware"])
        unlink.assert_called_once_with(ramdisk._manifest_path())

    def test_background_workers_use_noninteractive_sudo_after_foreground_validation(self):
        command = ["/usr/bin/mount", "-t", "tmpfs"]
        with mock.patch.object(
            ramdisk.os, "geteuid", return_value=1000, create=True
        ), mock.patch.object(
            ramdisk, "_trusted_system_binary", return_value="/usr/bin/sudo"
        ):
            foreground = ramdisk._privileged(command, {})
            with ramdisk._noninteractive_privilege():
                background = ramdisk._privileged(command, {})

        self.assertEqual(foreground[:2], ["/usr/bin/sudo", "--"])
        self.assertEqual(background[:3], ["/usr/bin/sudo", "-n", "--"])

    def test_sudo_keepalive_never_prompts(self):
        stop = mock.Mock()
        stop.is_set.return_value = False
        stop.wait.return_value = True
        completed = subprocess.CompletedProcess(
            ["/usr/bin/sudo", "-n", "-v"],
            0,
        )
        with mock.patch.object(
            ramdisk.subprocess,
            "run",
            return_value=completed,
        ) as run:
            ramdisk._sudo_ticket_keepalive(
                stop,
                "/usr/bin/sudo",
                interval=0.01,
            )

        run.assert_called_once_with(
            ["/usr/bin/sudo", "-n", "-v"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=5.0,
        )
        stop.wait.assert_called_once_with(0.01)

    def test_failed_sudo_keepalive_requests_cancellable_rollback(self):
        stop = mock.Mock()
        stop.is_set.return_value = False
        failure = threading.Event()
        cancel = threading.Event()
        completed = subprocess.CompletedProcess(
            ["/usr/bin/sudo", "-n", "-v"],
            1,
        )
        with mock.patch.object(
            ramdisk.subprocess,
            "run",
            return_value=completed,
        ):
            ramdisk._sudo_ticket_keepalive(
                stop,
                "/usr/bin/sudo",
                interval=0.01,
                failure_event=failure,
                cancel_event=cancel,
            )

        self.assertTrue(failure.is_set())
        self.assertTrue(cancel.is_set())
        stop.wait.assert_not_called()

    def test_sudo_authorization_is_checked_for_background_reuse(self):
        completed = subprocess.CompletedProcess(
            ["/usr/bin/sudo", "-n", "-v"],
            0,
        )
        with mock.patch.object(
            ramdisk.subprocess,
            "run",
            return_value=completed,
        ) as run:
            result = ramdisk._validate_noninteractive_sudo("/usr/bin/sudo")

        self.assertEqual(result.returncode, 0)
        run.assert_called_once_with(
            ["/usr/bin/sudo", "-n", "-v"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )



if __name__ == "__main__":
    unittest.main()
