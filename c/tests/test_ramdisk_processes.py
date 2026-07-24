"""RAM-disk managed-process launch and cleanup tests."""

if __package__:
    from .ramdisk_test_support import *  # noqa: F401,F403
else:
    from ramdisk_test_support import *  # noqa: F401,F403


class ManagedLaunchTest(unittest.TestCase):
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

            def poll(self):
                return None

            def wait(self, timeout=None):
                return 0

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
                ramdisk.subprocess, "Popen", return_value=FakeProcess()
            ), mock.patch.object(
                ramdisk, "_proc_identity", return_value=identity
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

            def poll(self):
                return None

            def wait(self, timeout=None):
                return 0

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
                ramdisk.subprocess, "Popen", return_value=FakeProcess()
            ), mock.patch.object(
                ramdisk, "_proc_identity", return_value=identity
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

