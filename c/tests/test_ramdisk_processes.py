"""RAM-disk managed-process launch and cleanup tests."""

import linecache

if __package__:
    from .ramdisk_test_support import *  # noqa: F401,F403
else:
    from ramdisk_test_support import *  # noqa: F401,F403

from ramdisk_support import lifecycle as lifecycle_support
from ramdisk_support import state as state_support


class ManagedLaunchTest(unittest.TestCase):
    def _exercise_launch_line_interrupt(self, source_fragment):
        class FakeSocket:
            def bind(self, address):
                pass

            def close(self):
                pass

        class LiveProcess:
            pid = 7298

            def poll(self):
                return None

            def wait(self, timeout=None):
                raise subprocess.TimeoutExpired("engine", timeout)

        nonce = "6" * 48
        merge_id = "7" * 32
        process = LiveProcess()
        identity = {
            "pid": process.pid,
            "pgid": process.pid,
            "uid": host_uid(),
            "starttime": 17298,
            "nonce": nonce,
        }
        snapshots = []
        trace_hits = []

        def trace_interrupt(frame, event, arg):
            if (
                event == "line"
                and frame.f_code is lifecycle_support.start.__code__
                and source_fragment
                in linecache.getline(frame.f_code.co_filename, frame.f_lineno)
            ):
                trace_hits.append(frame.f_lineno)
                sys.settrace(None)
                raise KeyboardInterrupt("line-boundary interruption")
            return trace_interrupt

        with ModelFixture() as fixture, canonical_temporary_directory() as state:
            plan = ramdisk.build_plan(
                plan_args(fixture.root), hardware=hardware_fixture()
            )
            with mock.patch.dict(
                os.environ, {"XDG_STATE_HOME": state}
            ):
                expected_state_root = ramdisk._state_root()
                expected_manifest_path = ramdisk._manifest_path()
                expected_benchmarks_path = ramdisk._benchmarks_path()
            plan["durable_state"] = {
                "root": expected_state_root,
                "manifest": expected_manifest_path,
                "benchmarks": expected_benchmarks_path,
            }
            mount = dict(plan["mounts"][0])
            mount.update(
                {
                    "ownership": "managed",
                    "identity": {"mount_id": 4, "device": "0:9"},
                }
            )
            manifest = {
                "version": ramdisk.MANIFEST_VERSION,
                "deployment_id": "8" * 32,
                "state": "ready",
                "base_port": 8000,
                "model_fingerprint": plan["model"]["fingerprint"],
                "plan": plan,
                "mounts": [mount],
                "processes": [],
                "ports": [],
            }

            def save(current):
                snapshot = json.loads(json.dumps(current))
                snapshots.append(snapshot)
                ramdisk._atomic_json(ramdisk._manifest_path(), snapshot)

            caught = None
            saved_manifest_path = None
            saved_state_root = None
            saved_benchmarks_path = None
            with mock.patch.dict(
                os.environ, {"XDG_STATE_HOME": state}
            ), mock.patch.multiple(
                ramdisk,
                _filesystem_for_path=mock.Mock(return_value="ext4"),
                _load_manifest=mock.Mock(return_value=manifest),
                _assert_effective_masks_unchanged=mock.Mock(),
                _assert_ready_mounts=mock.Mock(),
                _save_manifest=mock.Mock(side_effect=save),
                _admit_concurrent_runtimes=mock.Mock(),
                _recover_delta=mock.Mock(),
                _usage_read=mock.Mock(return_value={}),
                _usage_write=mock.Mock(),
                _proc_identity=mock.Mock(return_value=identity),
                _wait_managed_ready=mock.Mock(),
                _process_matches=mock.Mock(
                    return_value=(True, "running", identity)
                ),
                _terminate_verified_group=mock.Mock(
                    return_value="process group survived SIGKILL"
                ),
                _terminate_direct_child=mock.Mock(
                    return_value="direct child survived SIGKILL"
                ),
                _group_alive=mock.Mock(return_value=True),
                _track_managed_child=mock.Mock(),
                _forget_managed_child=mock.Mock(),
                _merge_usage=mock.Mock(),
            ), mock.patch.object(
                ramdisk.socket,
                "socket",
                side_effect=lambda *args, **kwargs: FakeSocket(),
            ), mock.patch.object(
                ramdisk.subprocess, "Popen", return_value=process
            ), mock.patch.object(
                ramdisk.secrets,
                "token_hex",
                side_effect=lambda size: nonce if size == 24 else merge_id,
            ):
                try:
                    sys.settrace(trace_interrupt)
                    ramdisk.start.__wrapped__(
                        argparse.Namespace(base_port=None),
                        cli_path=sys.executable,
                    )
                except BaseException as exc:
                    caught = exc
                finally:
                    sys.settrace(None)
                saved_manifest_path = ramdisk._manifest_path()
                saved_state_root = ramdisk._state_root()
                saved_benchmarks_path = ramdisk._benchmarks_path()

            load_error = None
            try:
                state_support._load_manifest(
                    required=True,
                    filesystem_for_path=lambda ignored: "ext4",
                    read_json=ramdisk._read_json,
                    manifest_path=lambda: saved_manifest_path,
                    state_root=lambda: saved_state_root,
                    benchmarks_path=lambda: saved_benchmarks_path,
                    assert_durable_state_dir=lambda path, plan=None: None,
                    uid_provider=host_uid,
                )
            except ramdisk.RamdiskError as exc:
                load_error = exc

        return manifest, snapshots, trace_hits, caught, load_error

    def _launch_authorities(self, manifest):
        return [
            (kind, entry["state_dir"])
            for kind, entries in (
                ("pending", manifest.get("pending_launches", [])),
                ("published", manifest.get("processes", [])),
                (
                    "retained",
                    manifest.get("recovery", {}).get(
                        "retained_processes", []
                    ),
                ),
            )
            for entry in entries
        ]

    def _exercise_prepublication_popen_outcome(
        self,
        *,
        popen_effect=None,
        popen_factory=None,
        cancel_after_pending=False,
        log_open_effect=None,
        terminate_direct_child_effect=None,
        group_alive_effect=None,
        merge_effect=None,
    ):
        class FakeSocket:
            def bind(self, address):
                pass

            def close(self):
                pass

        nonce = "d" * 48
        merge_id = "e" * 32
        cancel = threading.Event()
        snapshots = []

        with ModelFixture() as fixture, canonical_temporary_directory() as state:
            plan = ramdisk.build_plan(
                plan_args(fixture.root), hardware=hardware_fixture()
            )
            manifest = {
                "state": "ready",
                "base_port": 8000,
                "model_fingerprint": plan["model"]["fingerprint"],
                "plan": plan,
                "mounts": [dict(plan["mounts"][0])],
                "processes": [],
                "ports": [],
            }

            def save(current):
                snapshot = json.loads(json.dumps(current))
                snapshots.append(snapshot)
                if cancel_after_pending and snapshot.get("pending_launches"):
                    cancel.set()

            merge = mock.Mock(side_effect=merge_effect)
            if popen_factory is None:
                popen = mock.Mock(side_effect=popen_effect)
            else:
                self.assertIsNone(popen_effect)
                popen = popen_factory
            terminate_direct_child = mock.Mock(
                side_effect=terminate_direct_child_effect
            )
            group_alive = (
                mock.Mock(return_value=False)
                if group_alive_effect is None
                else mock.Mock(side_effect=group_alive_effect)
            )
            real_open = open
            caught = None
            with mock.patch.dict(
                os.environ, {"XDG_STATE_HOME": state}
            ), mock.patch.multiple(
                ramdisk,
                _filesystem_for_path=mock.Mock(return_value="ext4"),
                _load_manifest=mock.Mock(return_value=manifest),
                _assert_effective_masks_unchanged=mock.Mock(),
                _assert_ready_mounts=mock.Mock(),
                _save_manifest=mock.Mock(side_effect=save),
                _admit_concurrent_runtimes=mock.Mock(),
                _recover_delta=mock.Mock(),
                _usage_read=mock.Mock(return_value={}),
                _usage_write=mock.Mock(),
                _process_matches=mock.Mock(),
                _terminate_verified_group=mock.Mock(),
                _terminate_direct_child=terminate_direct_child,
                _group_alive=group_alive,
                _track_managed_child=mock.Mock(),
                _forget_managed_child=mock.Mock(),
                _merge_usage=merge,
            ), mock.patch.object(
                ramdisk.socket,
                "socket",
                side_effect=lambda *args, **kwargs: FakeSocket(),
            ), mock.patch.object(
                ramdisk.subprocess, "Popen", popen
            ), mock.patch.object(
                ramdisk.secrets,
                "token_hex",
                side_effect=lambda size: nonce if size == 24 else merge_id,
            ), mock.patch.object(
                lifecycle_support,
                "open",
                side_effect=(
                    log_open_effect
                    if log_open_effect is not None
                    else real_open
                ),
                create=True,
            ):
                try:
                    ramdisk.start.__wrapped__(
                        argparse.Namespace(base_port=None),
                        cli_path=sys.executable,
                        cancel_event=cancel,
                    )
                except BaseException as exc:
                    caught = exc

        return (
            manifest,
            snapshots,
            popen,
            merge,
            terminate_direct_child,
            group_alive,
            caught,
        )

    def test_launch_rollback_keeps_live_direct_child_when_proc_identity_is_inconclusive(self):
        class LiveProcess:
            pid = 7100

            def poll(self):
                return None

            def wait(self, timeout=None):
                raise subprocess.TimeoutExpired("engine", timeout)

        record = {"pid": 7100, "pgid": 7100}
        forget = mock.Mock()
        direct_terminate = mock.Mock(
            return_value="direct child PID 7100 survived SIGKILL"
        )
        failures, surviving = lifecycle_support._rollback_launched_children(
            [LiveProcess()],
            [record],
            process_matches=lambda ignored: (
                False,
                "identity-unavailable",
                None,
            ),
            group_alive=mock.Mock(return_value=True),
            track_managed_child=mock.Mock(),
            terminate_verified_group=lambda ignored: (
                "cannot revalidate managed process identity"
            ),
            terminate_direct_child=direct_terminate,
            forget_managed_child=forget,
        )

        self.assertEqual(surviving, {7100})
        self.assertIn("cannot revalidate", failures[0])
        self.assertIn("direct child is still alive", failures[0])
        direct_terminate.assert_called_once()
        forget.assert_not_called()

    def test_launch_rollback_discards_termination_failure_only_after_proven_absence(self):
        class ExitedProcess:
            pid = 7200

            def poll(self):
                return 17

            def wait(self, timeout=None):
                return 17

        forget = mock.Mock()
        failures, surviving = lifecycle_support._rollback_launched_children(
            [ExitedProcess()],
            [{"pid": 7200, "pgid": 7200}],
            process_matches=lambda ignored: (
                False,
                "not-running",
                None,
            ),
            group_alive=mock.Mock(return_value=False),
            track_managed_child=mock.Mock(),
            terminate_verified_group=lambda ignored: "late SIGKILL timeout",
            terminate_direct_child=mock.Mock(),
            forget_managed_child=forget,
        )

        self.assertEqual(failures, [])
        self.assertEqual(surviving, set())
        forget.assert_called_once_with(7200)

    def test_launch_rollback_retains_forked_engine_after_wrapper_exits(self):
        class ExitedWrapper:
            pid = 7250

            def poll(self):
                return 1

            def wait(self, timeout=None):
                return 1

        forget = mock.Mock()
        failures, surviving = lifecycle_support._rollback_launched_children(
            [ExitedWrapper()],
            [{"pid": 7250, "pgid": 7250}],
            process_matches=lambda ignored: (
                True,
                "running-group",
                {"pgid": 7250, "members": [{"pid": 7251}]},
            ),
            group_alive=mock.Mock(return_value=True),
            track_managed_child=mock.Mock(),
            terminate_verified_group=lambda ignored: (
                "process group 7250 survived SIGKILL"
            ),
            terminate_direct_child=mock.Mock(),
            forget_managed_child=forget,
        )

        self.assertEqual(surviving, {7250})
        self.assertIn("persisted process identity is still running", failures[0])
        forget.assert_not_called()

    def test_launch_rollback_retains_unpublished_group_after_wrapper_exits(self):
        class ExitedWrapper:
            pid = 7275

            def poll(self):
                return 1

            def wait(self, timeout=None):
                return 1

        context = {
            "state_dir": "/state/unpublished",
            "usage_baseline": {},
            "record": None,
        }
        group_alive = mock.Mock(return_value=True)
        forget = mock.Mock()
        failures, surviving = lifecycle_support._rollback_launched_children(
            [ExitedWrapper()],
            [],
            process_matches=mock.Mock(),
            group_alive=group_alive,
            track_managed_child=mock.Mock(),
            terminate_verified_group=mock.Mock(),
            terminate_direct_child=mock.Mock(return_value=None),
            forget_managed_child=forget,
            launch_contexts=[context],
        )

        self.assertEqual(surviving, {7275})
        self.assertIn(
            "direct-created process group 7275 is still alive",
            failures[0],
        )
        self.assertTrue(context["rollback_process_alive"])
        self.assertEqual(context["rollback_pid"], 7275)
        group_alive.assert_called_once_with(7275)
        forget.assert_not_called()

    def test_launch_rollback_treats_interrupted_group_scan_as_unproven(self):
        class ExitedWrapper:
            pid = 7280

            def poll(self):
                return 1

            def wait(self, timeout=None):
                return 1

        context = {
            "state_dir": "/state/interrupted-scan",
            "usage_baseline": {},
            "record": None,
        }
        failures, surviving = lifecycle_support._rollback_launched_children(
            [ExitedWrapper()],
            [],
            process_matches=mock.Mock(),
            group_alive=mock.Mock(side_effect=KeyboardInterrupt()),
            track_managed_child=mock.Mock(),
            terminate_verified_group=mock.Mock(),
            terminate_direct_child=mock.Mock(return_value=None),
            forget_managed_child=mock.Mock(),
            launch_contexts=[context],
        )

        self.assertEqual(surviving, {7280})
        self.assertIn(
            "could not establish direct-created process group 7280 absence",
            failures[0],
        )
        self.assertTrue(context["rollback_process_alive"])

    def test_pending_launch_is_durable_until_exact_process_promotion(self):
        class FakeSocket:
            def bind(self, address):
                pass

            def close(self):
                pass

        class LiveProcess:
            pid = 7290

            def poll(self):
                return None

            def wait(self, timeout=None):
                raise subprocess.TimeoutExpired("engine", timeout)

        nonce = "a" * 48
        merge_id = "b" * 32
        process = LiveProcess()
        identity = {
            "pid": process.pid,
            "pgid": process.pid,
            "uid": host_uid(),
            "starttime": 17290,
            "nonce": nonce,
        }
        snapshots = []
        successful = []

        def save(current):
            snapshot = json.loads(json.dumps(current))
            snapshots.append(snapshot)
            if snapshot.get("processes") and not snapshot.get(
                "pending_launches"
            ):
                raise OSError("exact process promotion write failed")
            successful.append(snapshot)

        def popen(*args, **kwargs):
            pending = successful[-1]["pending_launches"][0]
            self.assertEqual(successful[-1]["processes"], [])
            self.assertEqual(pending["nonce"], nonce)
            self.assertEqual(pending["usage_merge_id"], merge_id)
            self.assertEqual(pending["operation_id"], "start:" + merge_id)
            return process

        def proc_identity(_pid):
            # Popen has returned, but exact PID/PGID publication has not. A
            # hard crash here leaves the durable pending record intact.
            self.assertEqual(successful[-1]["processes"], [])
            self.assertEqual(
                successful[-1]["pending_launches"][0]["state_dir"],
                snapshots[-1]["pending_launches"][0]["state_dir"],
            )
            return identity

        with ModelFixture() as fixture, canonical_temporary_directory() as state:
            plan = ramdisk.build_plan(
                plan_args(fixture.root), hardware=hardware_fixture()
            )
            manifest = {
                "state": "ready",
                "base_port": 8000,
                "model_fingerprint": plan["model"]["fingerprint"],
                "plan": plan,
                "mounts": [dict(plan["mounts"][0])],
                "processes": [],
                "ports": [],
            }
            merge = mock.Mock()
            with mock.patch.dict(
                os.environ, {"XDG_STATE_HOME": state}
            ), mock.patch.multiple(
                ramdisk,
                _filesystem_for_path=mock.Mock(return_value="ext4"),
                _load_manifest=mock.Mock(return_value=manifest),
                _assert_effective_masks_unchanged=mock.Mock(),
                _assert_ready_mounts=mock.Mock(),
                _save_manifest=mock.Mock(side_effect=save),
                _admit_concurrent_runtimes=mock.Mock(),
                _recover_delta=mock.Mock(),
                _usage_read=mock.Mock(return_value={}),
                _usage_write=mock.Mock(),
                _proc_identity=mock.Mock(side_effect=proc_identity),
                _wait_managed_ready=mock.Mock(),
                _process_matches=mock.Mock(),
                _terminate_verified_group=mock.Mock(),
                _terminate_direct_child=mock.Mock(
                    return_value="direct child PID 7290 survived SIGKILL"
                ),
                _group_alive=mock.Mock(return_value=True),
                _track_managed_child=mock.Mock(),
                _forget_managed_child=mock.Mock(),
                _merge_usage=merge,
            ), mock.patch.object(
                ramdisk.socket,
                "socket",
                side_effect=lambda *args, **kwargs: FakeSocket(),
            ), mock.patch.object(
                ramdisk.subprocess, "Popen", side_effect=popen
            ), mock.patch.object(
                ramdisk.secrets,
                "token_hex",
                side_effect=lambda size: nonce if size == 24 else merge_id,
            ):
                with self.assertRaisesRegex(
                    ramdisk.RamdiskError,
                    "exact process promotion write failed.*direct child",
                ):
                    ramdisk.start.__wrapped__(
                        argparse.Namespace(base_port=None),
                        cli_path=sys.executable,
                    )

        merge.assert_not_called()
        self.assertEqual(manifest["state"], "error")
        self.assertEqual(manifest["processes"], [])
        self.assertEqual(manifest["pending_launches"], [])
        retained = manifest["recovery"]["retained_processes"][0]
        self.assertEqual(retained["pid"], process.pid)
        self.assertEqual(retained["usage_baseline"], {})
        self.assertEqual(retained["usage_merge_id"], merge_id)

    def test_cancel_after_pending_save_is_rechecked_before_popen(self):
        manifest, snapshots, popen, _merge, _terminate, _group, caught = (
            self._exercise_prepublication_popen_outcome(
                cancel_after_pending=True
            )
        )

        self.assertIsInstance(caught, ramdisk._OperationCancelled)
        self.assertTrue(
            any(snapshot.get("pending_launches") for snapshot in snapshots)
        )
        popen.assert_not_called()
        self.assertEqual(manifest["state"], "ready")
        self.assertEqual(manifest["pending_launches"], [])
        self.assertEqual(manifest["processes"], [])

    def test_pre_spawn_log_open_oserror_clears_pending_launch(self):
        manifest, snapshots, popen, _merge, _terminate, _group, caught = (
            self._exercise_prepublication_popen_outcome(
                log_open_effect=OSError("log open failed")
            )
        )

        self.assertIsInstance(caught, OSError)
        self.assertTrue(
            any(snapshot.get("pending_launches") for snapshot in snapshots)
        )
        popen.assert_not_called()
        self.assertEqual(manifest["state"], "error")
        self.assertEqual(manifest["pending_launches"], [])
        self.assertEqual(manifest["processes"], [])

    def test_mocked_popen_oserror_retains_unknown_without_inspected_attempt(self):
        manifest, snapshots, popen, merge, _terminate, _group, caught = (
            self._exercise_prepublication_popen_outcome(
                popen_effect=OSError("parent-side Popen failure")
            )
        )

        self.assertIsInstance(caught, ramdisk.RamdiskError)
        self.assertIn("process creation outcome is unknown", str(caught))
        self.assertIsInstance(caught.__cause__, OSError)
        self.assertTrue(
            any(snapshot.get("pending_launches") for snapshot in snapshots)
        )
        popen.assert_called_once()
        merge.assert_not_called()
        self.assertEqual(manifest["state"], "error")
        self.assertEqual(manifest["processes"], [])
        self.assertEqual(len(manifest["pending_launches"]), 1)

    def test_inspected_prefork_popen_exception_proves_child_absence(self):
        real_popen = subprocess.Popen

        class PreForkFailurePopen(real_popen):
            def _execute_child(self, *args, **kwargs):
                del args, kwargs
                raise OSError("inspected pre-fork failure")

        (
            manifest,
            snapshots,
            _popen,
            merge,
            terminate,
            group,
            caught,
        ) = self._exercise_prepublication_popen_outcome(
            popen_factory=PreForkFailurePopen
        )

        self.assertIsInstance(caught, OSError)
        self.assertEqual(str(caught), "inspected pre-fork failure")
        self.assertTrue(
            any(snapshot.get("pending_launches") for snapshot in snapshots)
        )
        terminate.assert_not_called()
        group.assert_not_called()
        merge.assert_called_once()
        self.assertEqual(manifest["state"], "error")
        self.assertEqual(manifest["processes"], [])
        self.assertEqual(manifest["pending_launches"], [])

    @unittest.skipUnless(
        sys.platform.startswith("linux") and os.name == "posix",
        "requires Linux POSIX Popen post-fork ordering",
    )
    def test_postfork_popen_exception_retains_and_reaps_exact_child(self):
        real_popen = subprocess.Popen
        attempts = []
        attempt_owned_devnull = []
        child_stdin_streams = []

        class PostForkFailurePopen(real_popen):
            def __init__(self, _command, **kwargs):
                child_stdin_streams.append(kwargs["stdin"])
                super().__init__(
                    [
                        sys.executable,
                        "-c",
                        "import time; time.sleep(30)",
                    ],
                    **kwargs,
                )

            def _close_pipe_fds(self, *args):
                super()._close_pipe_fds(*args)
                attempts.append(self)
                attempt_owned_devnull.append(hasattr(self, "_devnull"))
                raise OSError("parent-side post-fork failure")

        def reap_attempt():
            if not attempts:
                return
            process = attempts[-1]
            if process.poll() is None:
                process.kill()
            process.wait(timeout=2)

        self.addCleanup(reap_attempt)

        def terminate_direct_child(process):
            self.assertIs(process, attempts[-1])
            process.kill()
            return None

        (
            manifest,
            snapshots,
            _popen,
            merge,
            terminate,
            group,
            caught,
        ) = self._exercise_prepublication_popen_outcome(
            popen_factory=PostForkFailurePopen,
            terminate_direct_child_effect=terminate_direct_child,
        )

        self.assertEqual(len(attempts), 1)
        self.assertEqual(attempt_owned_devnull, [False])
        self.assertEqual(len(child_stdin_streams), 1)
        self.assertTrue(child_stdin_streams[0].closed)
        process = attempts[0]
        self.assertIsInstance(caught, OSError)
        self.assertEqual(str(caught), "parent-side post-fork failure")
        self.assertTrue(process._child_created)
        self.assertIsNotNone(process.returncode)
        self.assertTrue(
            any(snapshot.get("pending_launches") for snapshot in snapshots)
        )
        terminate.assert_called_once_with(process)
        group.assert_called_once_with(process.pid)
        merge.assert_called_once()
        self.assertEqual(manifest["state"], "error")
        self.assertEqual(manifest["processes"], [])
        self.assertEqual(manifest["pending_launches"], [])

    def test_async_popen_interruption_retains_outcome_unknown_pending_launch(self):
        manifest, snapshots, popen, merge, _terminate, _group, caught = (
            self._exercise_prepublication_popen_outcome(
                popen_effect=KeyboardInterrupt("asynchronous interrupt")
            )
        )

        self.assertIsInstance(caught, ramdisk.RamdiskError)
        self.assertIn("process creation outcome is unknown", str(caught))
        self.assertIsInstance(caught.__cause__, KeyboardInterrupt)
        self.assertTrue(
            any(snapshot.get("pending_launches") for snapshot in snapshots)
        )
        popen.assert_called_once()
        merge.assert_not_called()
        self.assertEqual(manifest["state"], "error")
        self.assertEqual(manifest["processes"], [])
        self.assertEqual(len(manifest["pending_launches"]), 1)
        pending = manifest["pending_launches"][0]
        self.assertEqual(pending["operation_id"], "start:" + "e" * 32)
        self.assertEqual(pending["nonce"], "d" * 48)
        self.assertEqual(pending["usage_merge_id"], "e" * 32)

    def test_log_close_failure_rolls_back_returned_process_before_merge(self):
        events = []

        class ReturnedProcess:
            pid = 7295
            alive = True

            def poll(self):
                return None if self.alive else 0

            def wait(self, timeout=None):
                if self.alive:
                    raise subprocess.TimeoutExpired("engine", timeout)
                return 0

        class FaultyLog:
            def close(self):
                events.append("close")
                raise OSError("log close failed")

        process = ReturnedProcess()

        def terminate_direct_child(child):
            self.assertIs(child, process)
            events.append("terminate")
            child.alive = False

        def group_alive(pgid):
            self.assertEqual(pgid, process.pid)
            self.assertFalse(process.alive)
            events.append("absence-proven")
            return False

        def merge_usage(*args, **kwargs):
            self.assertIn("absence-proven", events)
            events.append("merge")

        (
            manifest,
            _snapshots,
            popen,
            merge,
            terminate,
            group,
            caught,
        ) = self._exercise_prepublication_popen_outcome(
            popen_effect=lambda *args, **kwargs: process,
            log_open_effect=lambda *args, **kwargs: FaultyLog(),
            terminate_direct_child_effect=terminate_direct_child,
            group_alive_effect=group_alive,
            merge_effect=merge_usage,
        )

        self.assertIsInstance(caught, OSError)
        popen.assert_called_once()
        terminate.assert_called_once_with(process)
        group.assert_called_once_with(process.pid)
        merge.assert_called_once()
        self.assertLess(events.index("terminate"), events.index("merge"))
        self.assertLess(events.index("absence-proven"), events.index("merge"))
        self.assertEqual(manifest["pending_launches"], [])
        self.assertEqual(manifest["processes"], [])

    def test_interrupt_after_handle_registration_keeps_one_loadable_authority(self):
        manifest, _snapshots, hits, caught, load_error = (
            self._exercise_launch_line_interrupt(
                'context["spawn_outcome"] = "created"'
            )
        )

        self.assertTrue(hits)
        self.assertIsInstance(caught, ramdisk.RamdiskError)
        self.assertIsNone(load_error)
        authorities = self._launch_authorities(manifest)
        self.assertEqual(len(authorities), 1)
        self.assertEqual(authorities[0][0], "retained")

    def test_interrupt_after_process_publication_keeps_published_authority_only(self):
        manifest, _snapshots, hits, caught, load_error = (
            self._exercise_launch_line_interrupt("records.append(record)")
        )

        self.assertTrue(hits)
        self.assertIsInstance(caught, ramdisk.RamdiskError)
        self.assertIsNone(load_error)
        authorities = self._launch_authorities(manifest)
        self.assertEqual(len(authorities), 1)
        self.assertEqual(authorities[0][0], "published")

    def test_failed_launch_never_merges_usage_while_direct_child_is_alive(self):
        nonce = "e" * 48

        class FakeSocket:
            def bind(self, address):
                pass

            def close(self):
                pass

        class LiveProcess:
            pid = 7300

            def poll(self):
                return None

            def wait(self, timeout=None):
                raise subprocess.TimeoutExpired("engine", timeout)

        identity = {
            "pid": 7300,
            "pgid": 7300,
            "uid": host_uid(),
            "starttime": 17300,
            "nonce": nonce,
        }
        with ModelFixture() as fixture, canonical_temporary_directory() as state:
            plan = ramdisk.build_plan(
                plan_args(fixture.root), hardware=hardware_fixture()
            )
            manifest = {
                "state": "ready",
                "base_port": 8000,
                "model_fingerprint": plan["model"]["fingerprint"],
                "plan": plan,
                "mounts": [dict(plan["mounts"][0])],
                "processes": [],
                "ports": [],
            }
            forget = mock.Mock()
            merge = mock.Mock()
            with mock.patch.dict(
                os.environ, {"XDG_STATE_HOME": state}
            ), mock.patch.multiple(
                ramdisk,
                _filesystem_for_path=mock.Mock(return_value="ext4"),
                _load_manifest=mock.Mock(return_value=manifest),
                _assert_effective_masks_unchanged=mock.Mock(),
                _assert_ready_mounts=mock.Mock(),
                _save_manifest=mock.Mock(),
                _admit_concurrent_runtimes=mock.Mock(),
                _recover_delta=mock.Mock(),
                _usage_read=mock.Mock(return_value={}),
                _usage_write=mock.Mock(),
                _proc_identity=mock.Mock(return_value=identity),
                _wait_managed_ready=mock.Mock(
                    side_effect=ramdisk.RamdiskError("not ready")
                ),
                _process_matches=mock.Mock(
                    return_value=(False, "identity-unavailable", None)
                ),
                _terminate_verified_group=mock.Mock(
                    return_value="could not revalidate process identity"
                ),
                _terminate_direct_child=mock.Mock(
                    return_value="direct child PID 7300 survived SIGKILL"
                ),
                _track_managed_child=mock.Mock(),
                _forget_managed_child=forget,
                _merge_usage=merge,
            ), mock.patch.object(
                ramdisk.socket,
                "socket",
                side_effect=lambda *args, **kwargs: FakeSocket(),
            ), mock.patch.object(
                ramdisk.subprocess, "Popen", return_value=LiveProcess()
            ), mock.patch.object(
                ramdisk.secrets, "token_hex", return_value=nonce
            ):
                with self.assertRaisesRegex(
                    ramdisk.RamdiskError,
                    "not ready.*direct child is still alive",
                ):
                    ramdisk.start.__wrapped__(
                        argparse.Namespace(base_port=None),
                        cli_path=sys.executable,
                    )

        merge.assert_not_called()
        forget.assert_not_called()
        self.assertEqual(manifest["state"], "error")
        self.assertNotIn("stopped_at", manifest["processes"][0])
        self.assertNotIn("usage_merged_at", manifest["processes"][0])
        self.assertIn("direct child is still alive", manifest["processes"][0]["stop_error"])
        self.assertFalse(
            manifest.get("recovery", {}).get("retained_processes", [])
        )

        # A published survivor stays in the normal process list so a fresh
        # invocation of stop can reach its verified group terminator instead
        # of deadlocking behind unpublished-process recovery.
        reloaded = json.loads(json.dumps(manifest))
        process_matches = mock.Mock(
            side_effect=[
                (True, "running", dict(identity)),
                (False, "not-running", None),
            ]
        )
        terminate = mock.Mock(return_value=None)
        merge_after_stop = mock.Mock()
        with mock.patch.multiple(
            ramdisk,
            _load_manifest=mock.Mock(return_value=reloaded),
            _process_matches=process_matches,
            _managed_child_liveness=mock.Mock(return_value=False),
            _save_manifest=mock.Mock(),
            _terminate_verified_group=terminate,
            _merge_usage=merge_after_stop,
        ):
            stopped = ramdisk.stop.__wrapped__()

        terminate.assert_called_once_with(reloaded["processes"][0])
        merge_after_stop.assert_called_once()
        self.assertEqual(stopped["state"], "stopped")
        self.assertNotIn("stop_error", stopped["processes"][0])

    def test_full_mode_start_refuses_wrong_usage_identity_before_seed_or_spawn(self):
        class FakeSocket:
            def bind(self, address):
                pass

            def close(self):
                pass

        with ModelFixture() as fixture, canonical_temporary_directory() as state:
            plan = ramdisk.build_plan(
                plan_args(fixture.root, mode="full"),
                hardware=hardware_fixture(),
            )
            manifest = {
                "state": "ready",
                "base_port": 8000,
                "model_fingerprint": plan["model"]["fingerprint"],
                "plan": plan,
                "mounts": [dict(plan["mounts"][0])],
                "processes": [],
                "ports": [],
            }
            with mock.patch.dict(
                os.environ, {"XDG_STATE_HOME": state}
            ), mock.patch.object(
                ramdisk, "_filesystem_for_path", return_value="ext4"
            ), mock.patch.object(
                ramdisk, "_load_manifest", return_value=manifest
            ), mock.patch.object(
                ramdisk, "_assert_effective_masks_unchanged"
            ), mock.patch.object(
                ramdisk, "_assert_ready_mounts"
            ), mock.patch.object(
                ramdisk, "_save_manifest"
            ), mock.patch.object(
                ramdisk, "_admit_concurrent_runtimes"
            ), mock.patch.object(
                ramdisk.socket,
                "socket",
                side_effect=lambda *args, **kwargs: FakeSocket(),
            ), mock.patch.object(
                ramdisk, "_usage_write"
            ) as usage_write, mock.patch.object(
                ramdisk.subprocess, "Popen"
            ) as popen:
                for header, message in (
                    ("-1 1 2\n-2 1 1\n", "engine identity"),
                    (
                        "-1 9 2\n-2 1 3815245270\n",
                        "dimensions",
                    ),
                ):
                    with self.subTest(message=message):
                        (fixture.root / ".coli_usage").write_text(
                            header + "0 1 10\n",
                            encoding="utf-8",
                        )
                        with self.assertRaisesRegex(
                            ramdisk.RamdiskError,
                            message,
                        ):
                            ramdisk.start.__wrapped__(
                                argparse.Namespace(base_port=None),
                                cli_path=sys.executable,
                            )

        usage_write.assert_not_called()
        popen.assert_not_called()
        self.assertEqual(manifest["state"], "ready")

    def test_start_stop_preserve_headered_usage_identity(self):
        engine_id = 3815245270
        usage_header = "-1 1 2\n-2 1 %d\n" % engine_id

        class FakeSocket:
            def bind(self, address):
                pass

            def close(self):
                pass

        class FakeProcess:
            pid = 4300

            def __init__(self):
                self.returncode = None

            def poll(self):
                return self.returncode

        nonce = "a" * 48
        identity = {
            "pid": 4300,
            "pgid": 4300,
            "uid": host_uid(),
            "starttime": 14300,
            "nonce": nonce,
        }
        process = FakeProcess()
        with ModelFixture() as fixture, canonical_temporary_directory() as state:
            canonical = fixture.root / ".coli_usage"
            canonical.write_text(
                usage_header + "0 1 10\n",
                encoding="utf-8",
            )
            plan = ramdisk.build_plan(
                plan_args(fixture.root), hardware=hardware_fixture()
            )
            manifest = {
                "state": "ready",
                "base_port": 8000,
                "model_fingerprint": plan["model"]["fingerprint"],
                "plan": plan,
                "mounts": [dict(plan["mounts"][0])],
                "processes": [],
                "ports": [],
            }
            with mock.patch.dict(
                os.environ, {"XDG_STATE_HOME": state}
            ), mock.patch.object(
                ramdisk, "_filesystem_for_path", return_value="ext4"
            ), mock.patch.object(
                ramdisk, "_load_manifest", return_value=manifest
            ), mock.patch.object(
                ramdisk, "_assert_effective_masks_unchanged"
            ), mock.patch.object(
                ramdisk, "_assert_ready_mounts"
            ), mock.patch.object(
                ramdisk, "_save_manifest"
            ), mock.patch.object(
                ramdisk, "_admit_concurrent_runtimes"
            ), mock.patch.object(
                ramdisk.socket,
                "socket",
                side_effect=lambda *args, **kwargs: FakeSocket(),
            ), mock.patch.object(
                ramdisk.subprocess, "Popen", return_value=process
            ), mock.patch.object(
                ramdisk, "_proc_identity", return_value=identity
            ), mock.patch.object(
                ramdisk, "_wait_managed_ready"
            ), mock.patch.object(
                ramdisk,
                "_process_matches",
                return_value=(False, "not-running", None),
            ), mock.patch.object(
                ramdisk.secrets,
                "token_hex",
                side_effect=lambda size: "a" * (size * 2),
            ):
                launched = ramdisk.start.__wrapped__(
                    argparse.Namespace(base_port=None),
                    cli_path=sys.executable,
                )
                record = launched["processes"][0]
                state_usage = Path(record["state_dir"]) / ".coli_usage"
                self.assertTrue(
                    state_usage.read_text(encoding="utf-8").startswith(
                        usage_header
                    )
                )
                current = ramdisk._usage_read(str(state_usage))
                current["0:1"] = 12
                ramdisk._usage_write(str(state_usage), current)
                process.returncode = 0
                stopped = ramdisk.stop.__wrapped__()

            self.addCleanup(ramdisk._forget_managed_child, 4300)
            merged = ramdisk._usage_read(str(canonical))
            self.assertEqual(stopped["state"], "stopped")
            self.assertEqual(merged["0:1"], 12)
            self.assertEqual(merged["-1:1"], 2)
            self.assertEqual(merged["-2:1"], engine_id)

    def test_per_node_launch_forces_durable_kv_and_node_local_core_counts(self):
        captures = []
        nonce = "a" * 48

        class FakeSocket:
            def bind(self, address):
                pass

            def close(self):
                pass

        class FakeProcess:
            next_pid = 4100

            def __init__(self):
                type(self).next_pid += 1
                self.pid = type(self).next_pid

            def poll(self):
                return None

        def popen(command, **kwargs):
            process = FakeProcess()
            captures.append(
                {
                    "command": list(command),
                    "environment": dict(kwargs["env"]),
                    "pid": process.pid,
                }
            )
            return process

        def identity(pid):
            return {
                "pid": pid,
                "pgid": pid,
                "uid": host_uid(),
                "starttime": 1000 + pid,
                "nonce": nonce,
            }

        with ModelFixture() as fixture, canonical_temporary_directory() as state:
            hardware = hardware_fixture(nodes=2)
            set_asymmetric_node_cores(hardware)
            plan = ramdisk.build_plan(
                plan_args(fixture.root, topology="per-node"), hardware=hardware
            )
            manifest = {
                "state": "ready",
                "base_port": 8100,
                "model_fingerprint": plan["model"]["fingerprint"],
                "plan": plan,
                "mounts": [dict(item) for item in plan["mounts"]],
                "processes": [],
                "best_runtime": {
                    "per-node": {
                        "variant": "partial_direct",
                        "knobs": {
                            "PIPE": 1,
                            "OMP_NUM_THREADS": 3,
                            "OMP_PROC_BIND": "spread",
                        },
                    }
                },
            }
            with mock.patch.dict(
                os.environ,
                {
                    "XDG_STATE_HOME": state,
                    "KVSAVE": "0",
                    "COLI_NO_OMP_TUNE": "1",
                    "COLI_OMP_TUNED": "1",
                },
            ), mock.patch.object(
                ramdisk, "_filesystem_for_path", return_value="ext4"
            ), mock.patch.object(ramdisk, "_load_manifest", return_value=manifest), mock.patch.object(
                ramdisk, "_assert_ready_mounts"
            ), mock.patch.object(ramdisk, "_save_manifest"), mock.patch.object(
                ramdisk, "_admit_concurrent_runtimes"
            ) as admit, mock.patch.object(ramdisk, "_recover_delta"), mock.patch.object(
                ramdisk, "_usage_read", return_value={}
            ), mock.patch.object(ramdisk, "_usage_write"), mock.patch.object(
                ramdisk, "_fresh_user_binary", return_value="/usr/bin/numactl"
            ), mock.patch.object(
                ramdisk.socket, "socket", side_effect=lambda *args, **kwargs: FakeSocket()
            ), mock.patch.object(ramdisk.subprocess, "Popen", side_effect=popen), mock.patch.object(
                ramdisk, "_proc_identity", side_effect=identity
            ), mock.patch.object(ramdisk, "_wait_managed_ready"), mock.patch.object(
                ramdisk.secrets, "token_hex", return_value=nonce
            ):
                result = ramdisk.start.__wrapped__(
                    argparse.Namespace(base_port=None), cli_path=sys.executable
                )
        for launch in captures:
            self.addCleanup(ramdisk._forget_managed_child, launch["pid"])

        self.assertEqual(result["state"], "running")
        admit.assert_called_once_with(plan, manifest["mounts"], benchmark=False)
        self.assertEqual(len(captures), 2)
        for index, (expected_cores, expected_cpus) in enumerate(
            ((3, "0-2"), (5, "3-7"))
        ):
            launch = captures[index]
            environment = launch["environment"]
            self.assertEqual(environment["KVSAVE"], "1")
            self.assertEqual(environment["PROF"], "1")
            self.assertEqual(environment["PIPE"], "1")
            self.assertEqual(environment["OMP_NUM_THREADS"], str(expected_cores))
            self.assertEqual(environment["OMP_PROC_BIND"], "spread")
            self.assertEqual(environment["COLI_NUMA"], "0")
            self.assertNotIn("COLI_NO_OMP_TUNE", environment)
            self.assertNotIn("COLI_OMP_TUNED", environment)
            self.assertEqual(environment["COLI_NUMA_NODES"], str(index))
            self.assertEqual(environment["COLI_CPU_AFFINITY"], expected_cpus)
            self.assertEqual(
                launch["command"][:3],
                [
                    "/usr/bin/numactl",
                    "--physcpubind=%s" % expected_cpus,
                    "--membind=%d" % index,
                ],
            )
            self.assertTrue(environment["COLI_STATE_DIR"].endswith("node-%d" % index))
        self.assertEqual([record["port"] for record in result["processes"]], [8100, 8101])
        self.assertEqual(result["base_port"], 8100)

    def test_gpu_plan_launch_applies_reviewed_devices_and_mmap_path(self):
        captures = []
        nonce = "d" * 48

        class FakeSocket:
            def bind(self, address):
                pass

            def close(self):
                pass

        class FakeProcess:
            pid = 4200

            def poll(self):
                return None

        def popen(command, **kwargs):
            captures.append(dict(kwargs["env"]))
            return FakeProcess()

        identity = {
            "pid": 4200,
            "pgid": 4200,
            "uid": host_uid(),
            "starttime": 14200,
            "nonce": nonce,
        }
        with ModelFixture() as fixture, canonical_temporary_directory() as state:
            hardware = hardware_fixture()
            hardware["gpus"] = [
                {
                    "index": 2,
                    "name": "GPU 2",
                    "uuid": "GPU-test-2",
                    "pci_bus_id": "0000:41:00.0",
                    "numa_node": 0,
                    "locality": "resolved",
                    "total_bytes": 32 * ramdisk.GIB,
                    "free_bytes": 28 * ramdisk.GIB,
                }
            ]
            model = ramdisk.scan_model(str(fixture.root))
            result = ramdisk._resolve_preset(
                ramdisk.PRESET_GPU_FASTEST,
                plan_args(fixture.root),
                hardware=hardware,
                model=model,
                build_plan=ramdisk.build_plan,
                load_profile=ramdisk._load_profile,
                cuda_capable=True,
            )
            plan = result["plan"]
            manifest = {
                "state": "ready",
                "base_port": 8000,
                "model_fingerprint": plan["model"]["fingerprint"],
                "plan": plan,
                "mounts": [dict(plan["mounts"][0])],
                "processes": [],
            }
            with mock.patch.dict(
                os.environ,
                {
                    "XDG_STATE_HOME": state,
                    "COLI_GPUS": "7,8",
                    "CUDA_EXPERT_GB": "4",
                    "COLI_RAMMAP": "1",
                },
            ), mock.patch.object(
                ramdisk, "_filesystem_for_path", return_value="ext4"
            ), mock.patch.object(
                ramdisk, "_load_manifest", return_value=manifest
            ), mock.patch.object(
                ramdisk, "_assert_effective_masks_unchanged"
            ), mock.patch.object(
                ramdisk, "_assert_ready_mounts"
            ), mock.patch.object(
                ramdisk, "_save_manifest"
            ), mock.patch.object(
                ramdisk, "_admit_concurrent_runtimes"
            ), mock.patch.object(
                ramdisk, "_recover_delta"
            ), mock.patch.object(
                ramdisk, "_usage_read", return_value={}
            ), mock.patch.object(
                ramdisk, "_usage_write"
            ), mock.patch.object(
                ramdisk.socket,
                "socket",
                side_effect=lambda *args, **kwargs: FakeSocket(),
            ), mock.patch.object(
                ramdisk.subprocess,
                "Popen",
                side_effect=popen,
            ), mock.patch.object(
                ramdisk,
                "_proc_identity",
                return_value=identity,
            ), mock.patch.object(
                ramdisk, "_wait_managed_ready"
            ), mock.patch.object(
                ramdisk.secrets, "token_hex", return_value=nonce
            ):
                launched = ramdisk.start.__wrapped__(
                    argparse.Namespace(base_port=None),
                    cli_path=sys.executable,
                    engine_path=sys.executable,
                )
        self.addCleanup(ramdisk._forget_managed_child, 4200)

        self.assertEqual(launched["state"], "running")
        self.assertEqual(len(captures), 1)
        environment = captures[0]
        self.assertEqual(environment["COLI_CUDA"], "1")
        self.assertEqual(environment["COLI_GPU"], "0")
        self.assertNotIn("COLI_GPUS", environment)
        self.assertEqual(
            environment["CUDA_VISIBLE_DEVICES"],
            "GPU-test-2",
        )
        self.assertEqual(environment["CUDA_EXPERT_GB"], "auto")
        self.assertEqual(environment["REPIN"], "16")
        self.assertEqual(environment["COLI_MMAP"], "1")
        self.assertEqual(environment["COLI_RAMMAP"], "0")
        self.assertEqual(environment["COLI_RAM_PREFAULT"], "0")
        self.assertEqual(environment["PIN"], "auto")
        self.assertEqual(
            environment["COLI_ENGINE"],
            os.path.realpath(sys.executable),
        )
        self.assertEqual(
            launched["processes"][0]["accelerator_environment"]["COLI_GPU"],
            "0",
        )

    def test_clean_start_cancellation_restores_retryable_manifest(self):
        cancel = threading.Event()
        nonce = "c" * 48

        class FakeSocket:
            def bind(self, address):
                pass

            def close(self):
                pass

        class FakeProcess:
            pid = 6100

            def __init__(self):
                self.returncode = None

            def poll(self):
                return self.returncode

            def wait(self, timeout=None):
                self.returncode = 0
                return self.returncode

        def cancel_ready(*args, **kwargs):
            cancel.set()
            ramdisk._raise_if_cancelled(cancel)

        identity = {
            "pid": 6100,
            "pgid": 6100,
            "uid": host_uid(),
            "starttime": 16100,
            "nonce": nonce,
        }
        process = FakeProcess()
        with ModelFixture() as fixture, canonical_temporary_directory() as state:
            plan = ramdisk.build_plan(
                plan_args(fixture.root), hardware=hardware_fixture()
            )
            manifest = {
                "state": "ready",
                "base_port": 9000,
                "model_fingerprint": plan["model"]["fingerprint"],
                "plan": plan,
                "mounts": [dict(plan["mounts"][0])],
                "processes": [],
                "ports": [],
            }
            with mock.patch.dict(
                os.environ, {"XDG_STATE_HOME": state}
            ), mock.patch.object(
                ramdisk, "_filesystem_for_path", return_value="ext4"
            ), mock.patch.object(
                ramdisk, "_load_manifest", return_value=manifest
            ), mock.patch.object(ramdisk, "_assert_ready_mounts"), mock.patch.object(
                ramdisk, "_save_manifest"
            ), mock.patch.object(ramdisk, "_admit_concurrent_runtimes"), mock.patch.object(
                ramdisk, "_recover_delta"
            ), mock.patch.object(ramdisk, "_usage_read", return_value={}), mock.patch.object(
                ramdisk, "_usage_write"
            ), mock.patch.object(
                ramdisk.socket, "socket", side_effect=lambda *args, **kwargs: FakeSocket()
            ), mock.patch.object(
                ramdisk.subprocess, "Popen", return_value=process
            ) as popen, mock.patch.object(
                ramdisk,
                "_proc_identity",
                side_effect=lambda ignored: (
                    identity if process.poll() is None else None
                ),
            ), mock.patch.object(
                ramdisk, "_wait_managed_ready", side_effect=cancel_ready
            ), mock.patch.object(
                ramdisk, "_terminate_verified_group", return_value=None
            ), mock.patch.object(
                ramdisk, "_group_alive", return_value=False
            ), mock.patch.object(ramdisk, "_merge_usage"), mock.patch.object(
                ramdisk.secrets, "token_hex", return_value=nonce
            ):
                with self.assertRaises(ramdisk._OperationCancelled):
                    ramdisk.start.__wrapped__(
                        argparse.Namespace(base_port=None),
                        cli_path=sys.executable,
                        cancel_event=cancel,
                    )
                with self.assertRaises(ramdisk._OperationCancelled):
                    ramdisk.start.__wrapped__(
                        argparse.Namespace(base_port=None),
                        cli_path=sys.executable,
                        cancel_event=cancel,
                    )

        popen.assert_called_once()

        self.assertEqual(manifest["state"], "ready")
        self.assertEqual(manifest["base_port"], 9000)
        self.assertEqual(manifest["processes"], [])
        self.assertEqual(manifest["ports"], [])
        self.assertNotIn("launch_error", manifest)

    def test_launch_rollback_merges_every_context_when_manifest_saves_fail(self):
        nonce = "b" * 48

        class FakeSocket:
            def bind(self, address):
                pass

            def close(self):
                pass

        class FakeProcess:
            pid = 5100

            def __init__(self):
                self.returncode = None

            def poll(self):
                return self.returncode

            def wait(self, timeout=None):
                self.returncode = 0
                return self.returncode

        with ModelFixture() as fixture, canonical_temporary_directory() as state:
            plan = ramdisk.build_plan(plan_args(fixture.root), hardware=hardware_fixture())
            manifest = {
                "state": "ready",
                "model_fingerprint": plan["model"]["fingerprint"],
                "plan": plan,
                "mounts": [dict(plan["mounts"][0])],
                "processes": [],
            }

            def save(current):
                if current.get("state") == "error" or any(
                    record.get("usage_merge_id") for record in current.get("processes", [])
                ):
                    raise OSError("state filesystem full")

            identity = {
                "pid": 5100,
                "pgid": 5100,
                "uid": host_uid(),
                "starttime": 15100,
                "nonce": nonce,
            }
            process = FakeProcess()
            with mock.patch.dict(os.environ, {"XDG_STATE_HOME": state}), mock.patch.object(
                ramdisk, "_filesystem_for_path", return_value="ext4"
            ), mock.patch.object(
                ramdisk, "_load_manifest", return_value=manifest
            ), mock.patch.object(ramdisk, "_assert_ready_mounts"), mock.patch.object(
                ramdisk, "_save_manifest", side_effect=save
            ), mock.patch.object(ramdisk, "_admit_concurrent_runtimes"), mock.patch.object(
                ramdisk, "_recover_delta"
            ), mock.patch.object(ramdisk, "_usage_read", return_value={}), mock.patch.object(
                ramdisk, "_usage_write"
            ), mock.patch.object(
                ramdisk.socket, "socket", side_effect=lambda *args, **kwargs: FakeSocket()
            ), mock.patch.object(
                ramdisk.subprocess, "Popen", return_value=process
            ), mock.patch.object(
                ramdisk,
                "_proc_identity",
                side_effect=lambda ignored: (
                    identity if process.poll() is None else None
                ),
            ), mock.patch.object(
                ramdisk, "_wait_managed_ready", side_effect=ramdisk.RamdiskError("not ready")
            ), mock.patch.object(
                ramdisk, "_terminate_verified_group", return_value=None
            ), mock.patch.object(ramdisk, "_group_alive", return_value=False), mock.patch.object(
                ramdisk, "_merge_usage"
            ) as merge, mock.patch.object(ramdisk.secrets, "token_hex", return_value=nonce):
                with self.assertRaisesRegex(ramdisk.RamdiskError, "rollback/reporting errors"):
                    ramdisk.start.__wrapped__(
                        argparse.Namespace(base_port=8200), cli_path=sys.executable
                    )

        merge.assert_called_once()
        self.assertTrue(merge.call_args.kwargs["keep_journal"])
