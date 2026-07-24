"""RAM-disk presentation and interactive-console contract tests."""

if __package__:
    from .ramdisk_test_support import *  # noqa: F401,F403
else:
    from ramdisk_test_support import *  # noqa: F401,F403


class TuiPlacementContractTest(unittest.TestCase):
    def test_error_footer_advertises_the_available_recovery_actions(self):
        plan = {
            "mounts": [{"path": "/mnt/colibri-test", "node": None}],
            "blockers": [],
        }

        destroy_only = ramdisk._tui_idle_action_hint(
            0, plan, {"present": True, "state": "error", "processes": []}
        )
        stop_then_destroy = ramdisk._tui_idle_action_hint(
            0,
            plan,
            {"present": True, "state": "error", "processes": [{"pid": 123}]},
        )

        self.assertEqual(destroy_only, "[d] destroy")
        self.assertEqual(stop_then_destroy, "[x] stop  [d] destroy")

    def test_four_node_default_is_one_shared_model_and_one_engine(self):
        with ModelFixture() as fixture:
            plan = ramdisk.build_plan(
                plan_args(fixture.root), hardware=hardware_fixture(nodes=4)
            )

        placement = ramdisk._placement_summary(plan, base_port=8000)
        confirmation = ramdisk._prepare_confirmation(plan, base_port=8000)

        self.assertEqual(plan["topology"], "interleaved")
        self.assertEqual(plan["staging"]["replica_count"], 1)
        self.assertEqual(placement["copy_count"], 1)
        self.assertEqual(placement["engine_count"], 1)
        self.assertIn("Single shared model", placement["title"])
        self.assertIn("1 complete model copy", placement["cost"])
        self.assertIn("1 engine", placement["cost"])
        self.assertIn("spread across 4 NUMA nodes", placement["explanation"])
        self.assertIn("1 complete model copy", confirmation)
        self.assertIn("1 engine on port 8000", confirmation)

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
        confirmation = ramdisk._prepare_confirmation(replicated, base_port=8000)

        self.assertEqual(placement["copy_count"], 4)
        self.assertEqual(placement["engine_count"], 4)
        self.assertIn("Independent full-model replicas", placement["title"])
        self.assertIn("4 complete model copies", placement["cost"])
        self.assertIn("4 independent engines", placement["cost"])
        self.assertIn("ports 8000, 8001, 8002, 8003", placement["endpoints"])
        self.assertIn("replication, not model sharding", placement["explanation"])
        self.assertIn("4 complete model copies", confirmation)
        self.assertIn("4 independent engines", confirmation)
        self.assertIn("not sharding", confirmation)
        self.assertNotEqual(
            ramdisk._plan_confirmation_token(shared),
            ramdisk._plan_confirmation_token(replicated),
        )

    def test_prepare_rejects_a_plan_changed_after_tui_confirmation(self):
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

    def test_destroy_rejects_a_replacement_after_tui_confirmation(self):
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
        ) as unmount:
            with self.assertRaisesRegex(ramdisk.RamdiskError, "changed since review"):
                ramdisk.destroy.__wrapped__(
                    argparse.Namespace(yes=True),
                    expected_manifest_token=reviewed_token,
                )

        confirm.assert_not_called()
        unmount.assert_not_called()

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

    def test_minimum_viewport_pins_replica_warning_before_confirmation(self):
        with ModelFixture() as fixture:
            plan = ramdisk.build_plan(
                plan_args(fixture.root, topology="per-node"),
                hardware=hardware_fixture(nodes=4),
            )
        report = {"present": False, "state": "absent"}
        rows = ramdisk._tui_plan_rows(
            plan,
            report,
            active=False,
            base_port=8000,
            confirmation=ramdisk._prepare_confirmation(plan, 8000),
        )
        minimum_content = "\n".join(
            line for _, line in ramdisk._tui_wrap_rows(rows, 35)[:3]
        )

        self.assertIn("4 complete model copies", minimum_content)
        self.assertIn("4 independent engines", minimum_content)
        self.assertIn("replication, not sharding", minimum_content)

    def test_prepare_confirmation_cannot_coexist_with_scrolled_content(self):
        for requested_scroll in (1, 3, 100):
            with self.subTest(requested_scroll=requested_scroll):
                self.assertEqual(
                    ramdisk._tui_review_scroll("prepare", requested_scroll), 0
                )

        self.assertEqual(ramdisk._tui_review_scroll(None, 3), 3)
        self.assertEqual(ramdisk._tui_review_scroll("destroy", 3), 3)

    def test_running_plan_does_not_verify_a_dead_managed_process(self):
        with ModelFixture() as fixture:
            plan = ramdisk.build_plan(
                plan_args(fixture.root), hardware=hardware_fixture(nodes=2)
            )
        report = {
            "present": True,
            "state": "running",
            "deep_validation": False,
            "source_fingerprint_verified": None,
            "mounts": [
                {
                    "verified": True,
                    "namespace_verified": None,
                }
            ],
            "processes": [
                {
                    "running": False,
                    "verified": False,
                }
            ],
        }
        rendered = "\n".join(
            text for _, text in ramdisk._tui_plan_rows(plan, report, active=True)
        )

        self.assertIn("DEPLOYMENT NEEDS ATTENTION", rendered)
        self.assertNotIn("DEPLOYMENT VERIFIED", rendered)

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
            ramdisk, "_mount_at", side_effect=[None, actual, actual]
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
            ramdisk, "_mount_at", side_effect=[None, actual, actual]
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

    def test_tui_workers_use_noninteractive_sudo_after_foreground_validation(self):
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

    def test_interface_failure_reports_concurrent_cleanup_failure(self):
        interface_error = RuntimeError("display failed")
        cleanup_error = ramdisk.RamdiskError("rollback failed")
        worker_thread = mock.Mock()
        worker_thread.is_alive.return_value = False
        cancel_event = mock.Mock()
        ramdisk._tui_worker = {
            "cancelable": True,
            "cancel_event": cancel_event,
            "thread": worker_thread,
            "error": cleanup_error,
        }
        self.addCleanup(setattr, ramdisk, "_tui_worker", None)

        with mock.patch.dict(
            os.environ, {"COLI_RAMDISK_UI": "curses"}
        ), mock.patch("curses.wrapper", side_effect=interface_error):
            with self.assertRaisesRegex(
                ramdisk.RamdiskError,
                "interface failed while active operation cleanup also failed",
            ) as caught:
                ramdisk.launch_tui(argparse.Namespace())

        self.assertIs(caught.exception.__cause__, interface_error)
        cancel_event.set.assert_called_once_with()
        worker_thread.join.assert_called_once_with()
        self.assertIsNone(ramdisk._tui_worker)

    def test_minimum_width_settings_input_uses_a_full_entry_row(self):
        import curses

        class FakeScreen:
            def __init__(self):
                self.keys = iter(
                    [curses.KEY_RIGHT] * (len(ramdisk._TUI_SCREENS) - 1)
                    + [ord("o"), ord("q")]
                )
                self.getstr_call = None

            def timeout(self, milliseconds):
                pass

            def getmaxyx(self):
                return (8, 38)

            def erase(self):
                pass

            def addnstr(self, row, column, value, limit, attribute=0):
                pass

            def refresh(self):
                pass

            def getch(self):
                return next(self.keys)

            def move(self, row, column):
                pass

            def clrtoeol(self):
                pass

            def getstr(self, row, column, limit):
                self.getstr_call = (row, column, limit)
                return b""

        with ModelFixture() as fixture:
            hardware = hardware_fixture(nodes=2)
            model = ramdisk.scan_model(str(fixture.root))
            plan = ramdisk.build_plan(
                plan_args(fixture.root), hardware=hardware, model=model
            )
            report = {
                "present": False,
                "state": "absent",
                "mounts": [],
                "processes": [],
            }
            screen = FakeScreen()
            with mock.patch.object(
                ramdisk, "discover_hardware", return_value=hardware
            ), mock.patch.object(ramdisk, "scan_model", return_value=model), mock.patch.object(
                ramdisk, "build_plan", return_value=plan
            ), mock.patch.object(ramdisk, "status", return_value=report), mock.patch.object(
                curses, "curs_set"
            ), mock.patch.object(curses, "echo"), mock.patch.object(curses, "noecho"):
                result = ramdisk._tui(
                    screen, plan_args(fixture.root), "/fake/coli", "/fake/engine"
                )

        self.assertEqual(result, 0)
        self.assertEqual(screen.getstr_call[1], 2)
        self.assertGreaterEqual(screen.getstr_call[2], 34)

    def test_tui_has_no_single_key_path_into_replica_mode(self):
        class FakeScreen:
            def __init__(self):
                self.keys = iter((ord("t"), ord("q")))
                self.output = []

            def timeout(self, milliseconds):
                self.timeout_ms = milliseconds

            def getmaxyx(self):
                return (24, 100)

            def erase(self):
                pass

            def addnstr(self, row, column, value, limit, attribute=0):
                self.output.append(str(value)[:limit])

            def refresh(self):
                pass

            def getch(self):
                return next(self.keys)

        with ModelFixture() as fixture:
            hardware = hardware_fixture(nodes=4)
            model = ramdisk.scan_model(str(fixture.root))
            plan = ramdisk.build_plan(
                plan_args(fixture.root), hardware=hardware, model=model
            )
            report = {
                "present": False,
                "state": "absent",
                "mounts": [],
                "processes": [],
                "deep_validation": True,
            }
            observed = []

            def build(args, **kwargs):
                observed.append(args.topology)
                return plan

            screen = FakeScreen()
            with mock.patch.object(ramdisk, "discover_hardware", return_value=hardware), mock.patch.object(
                ramdisk, "scan_model", return_value=model
            ), mock.patch.object(ramdisk, "build_plan", side_effect=build), mock.patch.object(
                ramdisk, "status", return_value=report
            ):
                result = ramdisk._tui(
                    screen, plan_args(fixture.root), "/fake/coli", "/fake/engine"
                )

        self.assertEqual(result, 0)
        self.assertEqual(observed, ["interleaved"])
        rendered = "\n".join(screen.output)
        self.assertIn("Single shared model", rendered)
        self.assertIn("1 complete model copy", rendered)
