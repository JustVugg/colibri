"""RAM-disk mount, copy, and staged-namespace tests."""

if __package__:
    from .ramdisk_test_support import *  # noqa: F401,F403
else:
    from ramdisk_test_support import *  # noqa: F401,F403

from ramdisk_support import lifecycle as lifecycle_support
from ramdisk_support import mounts as mounts_support


class MountAndCopyTest(unittest.TestCase):
    def setUp(self):
        for name, value in (
            ("_ensure_busy_mount_scan_available", None),
            ("_busy_mount_references", []),
        ):
            patcher = mock.patch.object(ramdisk, name, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _stage_fixture_namespace(self, fixture, destination):
        for source in fixture.root.glob("*.safetensors"):
            target = Path(destination) / source.name
            shutil.copy2(source, target)
            target.chmod(0o400)

    def test_prepare_proves_cleanup_scan_before_confirmation_or_mutation(self):
        with ModelFixture() as fixture, mock.patch.object(
            ramdisk, "_filesystem_for_path", return_value="ext4"
        ):
            plan = ramdisk.build_plan(
                plan_args(fixture.root), hardware=hardware_fixture()
            )

        unavailable = ramdisk.RamdiskError(
            "install the psmisc package and retry"
        )
        with mock.patch.object(
            ramdisk, "_load_manifest", return_value=None
        ), mock.patch.object(
            ramdisk, "build_plan", return_value=plan
        ), mock.patch.object(
            ramdisk,
            "_ensure_busy_mount_scan_available",
            side_effect=unavailable,
        ) as ensure_scan, mock.patch.object(
            ramdisk, "_confirm"
        ) as confirm, mock.patch.object(
            ramdisk, "_save_manifest"
        ) as save, mock.patch.object(
            ramdisk, "_mount_tmpfs"
        ) as mount:
            with self.assertRaisesRegex(
                ramdisk.RamdiskError,
                "install the psmisc package",
            ):
                ramdisk.prepare.__wrapped__(
                    plan_args(fixture.root, yes=True),
                    display_plan=False,
                )

        ensure_scan.assert_called_once_with(
            plan["mount_root"],
            hardware=plan["hardware"],
        )
        confirm.assert_not_called()
        save.assert_not_called()
        mount.assert_not_called()

    def test_cgroup_headroom_uses_the_injected_discovery_service(self):
        discover = mock.Mock(
            return_value={"available_bytes": 1234, "error": None}
        )

        self.assertEqual(
            mounts_support._default_cgroup_available_memory(
                discover_cgroup_memory=discover,
            ),
            1234,
        )
        discover.assert_called_once_with()

        with mock.patch.object(
            ramdisk,
            "_discover_cgroup_memory",
            return_value={"available_bytes": 5678, "error": None},
        ):
            self.assertEqual(ramdisk._cgroup_available_memory(), 5678)

        with self.assertRaisesRegex(
            mounts_support.RamdiskError,
            "cannot validate cgroup memory headroom: unreadable",
        ):
            mounts_support._default_cgroup_available_memory(
                discover_cgroup_memory=lambda: {
                    "available_bytes": None,
                    "error": "unreadable",
                },
            )

    def test_mountinfo_preserves_noncontiguous_mpol_nodemask(self):
        line = (
            "36 25 0:32 / /mnt/colibri-ram rw,noatime - tmpfs tmpfs "
            "rw,noswap,nodev,nosuid,noexec,mode=700,huge=within_size,"
            "mpol=interleave:0-1,3\n"
        )
        # A still-open NamedTemporaryFile cannot be reopened by native Windows.
        # The parser is portable when fed a closed, byte-exact fixture.
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory) / "mountinfo"
            fixture.write_bytes(line.encode("utf-8"))
            parsed = ramdisk._mount_table(str(fixture))
        self.assertEqual(parsed[0]["super_options"][-1], "rw")
        self.assertIn("mpol=interleave:0-1,3", parsed[0]["super_options"])

    def test_mount_falls_back_from_within_size_to_advise_only_on_option_error(self):
        with ModelFixture() as fixture:
            plan = ramdisk.build_plan(plan_args(fixture.root), hardware=hardware_fixture())
        calls = []

        def run(command, **kwargs):
            calls.append(command)
            if len(calls) == 1:
                return subprocess.CompletedProcess(command, 32, "", "mount: invalid argument")
            return subprocess.CompletedProcess(command, 0, "", "")

        with mock.patch.object(
            ramdisk, "_trusted_system_binary", return_value="/bin/mount"
        ), mock.patch.object(ramdisk, "_run", side_effect=run), mock.patch.object(
            ramdisk, "_privileged", side_effect=lambda command, hardware: command
        ):
            ramdisk._mount_tmpfs(plan, plan["mounts"][0])
        self.assertIn("huge=within_size", calls[0][4])
        self.assertIn("huge=advise", calls[1][4])
        self.assertTrue(plan["mounts"][0]["effective_noswap"])

    def test_swappable_fallback_preserves_supported_within_size_thp(self):
        with ModelFixture() as fixture:
            plan = ramdisk.build_plan(
                plan_args(fixture.root, allow_swappable=True),
                hardware=hardware_fixture(),
            )
        calls = []

        def run(command, **kwargs):
            calls.append(command)
            if len(calls) <= 2:
                return subprocess.CompletedProcess(
                    command, 32, "", "mount: invalid argument"
                )
            return subprocess.CompletedProcess(command, 0, "", "")

        with mock.patch.object(
            ramdisk, "_trusted_system_binary", return_value="/bin/mount"
        ), mock.patch.object(ramdisk, "_run", side_effect=run), mock.patch.object(
            ramdisk, "_privileged", side_effect=lambda command, hardware: command
        ):
            ramdisk._mount_tmpfs(plan, plan["mounts"][0])
        self.assertEqual(len(calls), 3)
        self.assertIn("huge=within_size", calls[2][4])
        self.assertNotIn("noswap", calls[2][4].split(","))
        self.assertFalse(plan["mounts"][0]["effective_noswap"])

    def test_mount_keeps_private_tmpfs_over_reusable_underlying_directory(self):
        with ModelFixture() as fixture:
            plan = ramdisk.build_plan(plan_args(fixture.root), hardware=hardware_fixture())
        options = ramdisk._mount_option_list(plan, plan["mounts"][0])
        self.assertIn("mode=0700", options)
        self.assertIn("X-mount.mkdir=0755", options)

    def test_non_option_mount_error_does_not_retry_weaker_options(self):
        with ModelFixture() as fixture:
            plan = ramdisk.build_plan(plan_args(fixture.root), hardware=hardware_fixture())
        result = subprocess.CompletedProcess([], 1, "", "permission denied")
        with mock.patch.object(
            ramdisk, "_trusted_system_binary", return_value="/bin/mount"
        ), mock.patch.object(
            ramdisk, "_run", return_value=result
        ) as run, mock.patch.object(
            ramdisk, "_privileged", side_effect=lambda command, hardware: command
        ):
            with self.assertRaises(ramdisk.RamdiskError):
                ramdisk._mount_tmpfs(plan, plan["mounts"][0])
        self.assertEqual(run.call_count, 1)

    def test_unmount_uses_trusted_noncanonicalizing_util_linux_command(self):
        run = mock.Mock(
            return_value=subprocess.CompletedProcess([], 0, "", "")
        )
        hardware = hardware_fixture()

        mounts_support._umount_path(
            "/mnt/colibri-test",
            hardware,
            trusted_system_binary=mock.Mock(
                return_value="/usr/bin/umount"
            ),
            run=run,
            privileged=lambda command, ignored: command,
        )

        run.assert_called_once_with(
            [
                "/usr/bin/umount",
                "--no-canonicalize",
                "--",
                "/mnt/colibri-test",
            ]
        )

    @requires_linux_operational
    def test_interrupted_mount_helper_retains_pending_recovery_without_unmount(self):
        snapshots = []
        observed = {"mounted": False}
        with ModelFixture() as fixture, mock.patch.object(
            ramdisk, "_filesystem_for_path", return_value="ext4"
        ):
            plan = ramdisk.build_plan(
                plan_args(fixture.root), hardware=hardware_fixture()
            )
        mount = plan["mounts"][0]
        actual = {
            "mount_id": 44,
            "device": "0:44",
            "filesystem": "tmpfs",
            "source": "tmpfs",
        }
        interrupted = ramdisk._TuiTerminationSignal(signal.SIGTERM)

        def save(manifest):
            snapshots.append(json.loads(json.dumps(manifest)))

        def run(_command, **_kwargs):
            # The kernel mount completed, but delivery of the helper result was
            # interrupted before the lifecycle could persist its identity.
            observed["mounted"] = True
            raise interrupted

        def mount_at(_path):
            return actual if observed["mounted"] else None

        with mock.patch.object(
            ramdisk, "_load_manifest", return_value=None
        ), mock.patch.object(
            ramdisk, "build_plan", return_value=plan
        ), mock.patch.object(
            ramdisk, "_save_manifest", side_effect=save
        ), mock.patch.object(
            ramdisk, "_trusted_system_binary", return_value="/bin/mount"
        ), mock.patch.object(
            ramdisk, "_privileged", side_effect=lambda command, hardware: command
        ), mock.patch.object(
            ramdisk, "_run", side_effect=run
        ), mock.patch.object(
            ramdisk, "_mount_at", side_effect=mount_at
        ), mock.patch.object(
            ramdisk, "_validate_mount", return_value=actual
        ) as validate, mock.patch.object(
            ramdisk, "_mount_table", return_value=[]
        ), mock.patch.object(
            ramdisk, "_umount_path"
        ) as unmount:
            with self.assertRaisesRegex(
                ramdisk.RamdiskError,
                "pathname-only rollback",
            ):
                ramdisk.prepare.__wrapped__(
                    plan_args(fixture.root, yes=True), display_plan=False
                )

        self.assertTrue(observed["mounted"])
        validate.assert_not_called()
        unmount.assert_not_called()
        retained = snapshots[-1]
        self.assertEqual(retained["state"], "error")
        self.assertEqual(retained["recovery"]["state"], "attention-required")
        self.assertEqual(retained["recovery"]["retained_mounts"], [mount["path"]])
        self.assertEqual(retained["mounts"][0]["ownership"], "pending")
        self.assertEqual(retained["mounts"][0]["cleanup"]["state"], "retained")
        self.assertNotIn("identity", retained["mounts"][0])

    @requires_linux_operational
    def test_prepare_persists_pending_ownership_before_mount_helper(self):
        snapshots = []
        helper_saw_pending = []
        with ModelFixture() as fixture, mock.patch.object(
            ramdisk, "_filesystem_for_path", return_value="ext4"
        ):
            plan = ramdisk.build_plan(
                plan_args(fixture.root), hardware=hardware_fixture()
            )

        def save(manifest):
            snapshots.append(json.loads(json.dumps(manifest)))

        def mount_helper(_plan, mount):
            pending = snapshots[-1]["mounts"]
            helper_saw_pending.append(
                len(pending) == 1
                and pending[0]["path"] == mount["path"]
                and pending[0]["ownership"] == "pending"
                and pending[0]["operation_id"].endswith(":mount:0")
                and pending[0]["requested"]["filesystem"] == "tmpfs"
                and pending[0]["requested"]["source"] == "tmpfs"
                and "identity" not in pending[0]
            )
            raise ramdisk.RamdiskError("mount helper failed")

        with mock.patch.object(ramdisk, "_load_manifest", return_value=None), mock.patch.object(
            ramdisk, "build_plan", return_value=plan
        ), mock.patch.object(ramdisk, "_save_manifest", side_effect=save), mock.patch.object(
            ramdisk, "_mount_at", return_value=None
        ), mock.patch.object(ramdisk, "_mount_tmpfs", side_effect=mount_helper), mock.patch.object(
            ramdisk, "_umount_path"
        ) as unmount:
            with self.assertRaisesRegex(ramdisk.RamdiskError, "mount helper failed"):
                ramdisk.prepare.__wrapped__(
                    plan_args(fixture.root, yes=True), display_plan=False
                )

        self.assertEqual(helper_saw_pending, [True])
        unmount.assert_not_called()
        self.assertEqual(snapshots[-1]["state"], "error")
        self.assertEqual(
            snapshots[-1]["mounts"][0]["ownership"],
            "pending",
        )
        self.assertNotIn("identity", snapshots[-1]["mounts"][0])

    @requires_linux_operational
    def test_prepare_never_unmounts_identityless_successful_mount_by_path(self):
        snapshots = []
        with ModelFixture() as fixture, mock.patch.object(
            ramdisk, "_filesystem_for_path", return_value="ext4"
        ):
            plan = ramdisk.build_plan(
                plan_args(fixture.root), hardware=hardware_fixture()
            )

        def save(manifest):
            snapshots.append(json.loads(json.dumps(manifest)))

        with mock.patch.object(ramdisk, "_load_manifest", return_value=None), mock.patch.object(
            ramdisk, "build_plan", return_value=plan
        ), mock.patch.object(ramdisk, "_save_manifest", side_effect=save), mock.patch.object(
            ramdisk, "_mount_at", return_value=None
        ), mock.patch.object(ramdisk, "_mount_tmpfs"), mock.patch.object(
            ramdisk, "_umount_path"
        ) as unmount:
            with self.assertRaisesRegex(
                ramdisk.RamdiskError,
                "could not read its mount identity",
            ):
                ramdisk.prepare.__wrapped__(
                    plan_args(fixture.root, yes=True), display_plan=False
                )

        unmount.assert_not_called()
        retained = snapshots[-1]
        self.assertEqual(retained["state"], "error")
        self.assertEqual(retained["mounts"][0]["ownership"], "pending")
        self.assertEqual(
            retained["recovery"]["retained_mounts"],
            [plan["mounts"][0]["path"]],
        )

    @requires_linux_operational
    def test_prepare_promotes_only_the_exact_recorded_mount_identity(self):
        snapshots = []
        actual = {
            "mount_id": 19,
            "device": "0:19",
            "filesystem": "tmpfs",
            "source": "tmpfs",
        }
        with ModelFixture() as fixture, mock.patch.object(
            ramdisk, "_filesystem_for_path", return_value="ext4"
        ):
            plan = ramdisk.build_plan(
                plan_args(fixture.root), hardware=hardware_fixture()
            )

        def save(manifest):
            snapshots.append(json.loads(json.dumps(manifest)))

        def mount_helper(_plan, mount):
            mount["effective_thp"] = "advise"
            mount["effective_noswap"] = False

        with mock.patch.object(ramdisk, "_load_manifest", return_value=None), mock.patch.object(
            ramdisk, "build_plan", return_value=plan
        ), mock.patch.object(ramdisk, "_save_manifest", side_effect=save), mock.patch.object(
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
        ), mock.patch.object(ramdisk, "_mount_tmpfs", side_effect=mount_helper), mock.patch.object(
            ramdisk, "_validate_mount", return_value=actual
        ), mock.patch.object(
            ramdisk, "_populate_mount", side_effect=ramdisk.RamdiskError("copy failed")
        ), mock.patch.object(ramdisk, "_mount_table", return_value=[]), mock.patch.object(
            ramdisk, "_umount_path"
        ) as unmount:
            with self.assertRaisesRegex(ramdisk.RamdiskError, "copy failed"):
                ramdisk.prepare.__wrapped__(
                    plan_args(fixture.root, yes=True), display_plan=False
                )

        ownership_history = [
            snapshot["mounts"][0]["ownership"]
            for snapshot in snapshots
            if snapshot["mounts"]
        ]
        self.assertEqual(ownership_history[:3], ["pending", "identified", "managed"])
        identified = next(
            snapshot["mounts"][0]
            for snapshot in snapshots
            if snapshot["mounts"]
            and snapshot["mounts"][0]["ownership"] == "identified"
        )
        self.assertEqual(identified["identity"], actual)
        self.assertEqual(identified["effective_thp"], "advise")
        self.assertFalse(identified["effective_noswap"])
        unmount.assert_called_once_with(plan["mounts"][0]["path"], plan["hardware"])

    @requires_linux_operational
    def test_multi_mount_failure_preflights_all_before_any_unmount(self):
        snapshots = []
        observed = {}
        with ModelFixture() as fixture, mock.patch.object(
            ramdisk, "_filesystem_for_path", return_value="ext4"
        ):
            plan = ramdisk.build_plan(
                plan_args(fixture.root, topology="per-node"),
                hardware=hardware_fixture(nodes=2),
            )
        first, second = plan["mounts"]
        first_actual = {
            "mount_id": 31,
            "device": "0:31",
            "filesystem": "tmpfs",
            "source": "tmpfs",
        }

        def save(manifest):
            snapshots.append(json.loads(json.dumps(manifest)))

        def mount_tmpfs(_plan, mount):
            if mount["path"] == first["path"]:
                observed[mount["path"]] = first_actual
                return
            raise ramdisk.RamdiskError("second mount failed")

        def mount_at(path):
            return observed.get(path)

        def unmount(path, hardware):
            self.assertIs(hardware, plan["hardware"])
            observed.pop(path)

        with mock.patch.object(ramdisk, "_load_manifest", return_value=None), mock.patch.object(
            ramdisk, "build_plan", return_value=plan
        ), mock.patch.object(ramdisk, "_save_manifest", side_effect=save), mock.patch.object(
            ramdisk, "_mount_at", side_effect=mount_at
        ), mock.patch.object(ramdisk, "_mount_tmpfs", side_effect=mount_tmpfs), mock.patch.object(
            ramdisk, "_validate_mount", side_effect=lambda mount, ignored: observed[mount["path"]]
        ), mock.patch.object(ramdisk, "_mount_table", return_value=[]), mock.patch.object(
            ramdisk, "_umount_path", side_effect=unmount
        ) as unmount_mock:
            with self.assertRaisesRegex(ramdisk.RamdiskError, "second mount failed"):
                ramdisk.prepare.__wrapped__(
                    plan_args(fixture.root, yes=True), display_plan=False
                )

        unmount_mock.assert_not_called()
        recovery = snapshots[-1]["recovery"]
        self.assertEqual(recovery["released_mounts"], [])
        self.assertEqual(
            recovery["retained_mounts"],
            sorted([first["path"], second["path"]]),
        )
        by_path = {
            record["path"]: record
            for record in snapshots[-1]["mounts"]
        }
        self.assertEqual(by_path[first["path"]]["cleanup"]["state"], "retained")
        self.assertEqual(by_path[second["path"]]["ownership"], "pending")

    @requires_linux_operational
    def test_prepare_rollback_refuses_exact_mount_with_nested_child(self):
        snapshots = []
        actual = {
            "mount_id": 41,
            "device": "0:41",
            "filesystem": "tmpfs",
            "source": "tmpfs",
        }
        with ModelFixture() as fixture, mock.patch.object(
            ramdisk, "_filesystem_for_path", return_value="ext4"
        ):
            plan = ramdisk.build_plan(
                plan_args(fixture.root), hardware=hardware_fixture()
            )
        path = plan["mounts"][0]["path"]
        child = {"path": os.path.join(path, "foreign-child")}

        def save(manifest):
            snapshots.append(json.loads(json.dumps(manifest)))

        with mock.patch.object(ramdisk, "_load_manifest", return_value=None), mock.patch.object(
            ramdisk, "build_plan", return_value=plan
        ), mock.patch.object(ramdisk, "_save_manifest", side_effect=save), mock.patch.object(
            ramdisk, "_mount_at", side_effect=[None, actual, actual]
        ), mock.patch.object(ramdisk, "_mount_tmpfs"), mock.patch.object(
            ramdisk, "_validate_mount", return_value=actual
        ), mock.patch.object(
            ramdisk, "_populate_mount", side_effect=ramdisk.RamdiskError("copy failed")
        ), mock.patch.object(ramdisk, "_mount_table", return_value=[child]), mock.patch.object(
            ramdisk, "_umount_path"
        ) as unmount:
            with self.assertRaisesRegex(ramdisk.RamdiskError, "nested mount"):
                ramdisk.prepare.__wrapped__(
                    plan_args(fixture.root, yes=True), display_plan=False
                )

        unmount.assert_not_called()
        self.assertEqual(snapshots[-1]["recovery"]["retained_mounts"], [path])
        self.assertEqual(
            snapshots[-1]["mounts"][0]["cleanup"]["state"],
            "retained",
        )

    @requires_linux_operational
    def test_prepare_cleanup_runs_even_when_error_manifest_cannot_be_saved(self):
        snapshots = []
        with ModelFixture() as fixture, mock.patch.object(
            ramdisk, "_filesystem_for_path", return_value="ext4"
        ):
            plan = ramdisk.build_plan(
                plan_args(fixture.root), hardware=hardware_fixture()
            )

        def save(manifest):
            snapshots.append(json.loads(json.dumps(manifest)))
            if len(snapshots) > 1:
                raise OSError("state full")

        with mock.patch.object(ramdisk, "_load_manifest", return_value=None), mock.patch.object(
            ramdisk, "build_plan", return_value=plan
        ), mock.patch.object(
            ramdisk, "_save_manifest", side_effect=save
        ), mock.patch.object(
            ramdisk, "_mount_at", return_value=None
        ), mock.patch.object(ramdisk, "_mount_tmpfs") as mount, mock.patch.object(
            ramdisk, "_mount_table", return_value=[]
        ), mock.patch.object(
            ramdisk, "_umount_path"
        ) as unmount:
            with self.assertRaisesRegex(ramdisk.RamdiskError, "rollback/reporting errors"):
                ramdisk.prepare.__wrapped__(
                    plan_args(fixture.root, yes=True), display_plan=False
                )
        mount.assert_not_called()
        unmount.assert_not_called()
        self.assertEqual(
            snapshots[-1]["recovery"]["retained_mounts"],
            [],
        )
        self.assertEqual(snapshots[-1]["recovery"]["state"], "clean")

    def test_failed_identified_save_does_not_authorize_preparation_unmount(self):
        snapshots = []
        actual = {
            "mount_id": 51,
            "device": "0:51",
            "filesystem": "tmpfs",
            "source": "tmpfs",
        }
        with ModelFixture() as fixture, mock.patch.object(
            ramdisk, "_filesystem_for_path", return_value="ext4"
        ):
            plan = ramdisk.build_plan(
                plan_args(fixture.root), hardware=hardware_fixture()
            )

        def save(manifest):
            snapshots.append(json.loads(json.dumps(manifest)))
            if len(snapshots) >= 3:
                raise OSError("durable manifest unavailable")

        with mock.patch.object(
            ramdisk, "_load_manifest", return_value=None
        ), mock.patch.object(
            ramdisk, "build_plan", return_value=plan
        ), mock.patch.object(
            ramdisk, "_save_manifest", side_effect=save
        ), mock.patch.object(
            ramdisk, "_mount_at", side_effect=[None, actual, actual]
        ), mock.patch.object(
            ramdisk, "_mount_tmpfs"
        ), mock.patch.object(
            ramdisk, "_validate_mount", return_value=actual
        ) as validate, mock.patch.object(
            ramdisk, "_mount_table", return_value=[]
        ), mock.patch.object(
            ramdisk, "_busy_mount_references", return_value=[]
        ), mock.patch.object(
            ramdisk, "_umount_path"
        ) as unmount:
            with self.assertRaisesRegex(
                ramdisk.RamdiskError,
                "durable manifest unavailable",
            ):
                ramdisk.prepare.__wrapped__(
                    plan_args(fixture.root, yes=True),
                    display_plan=False,
                )

        validate.assert_not_called()
        unmount.assert_not_called()
        self.assertEqual(snapshots[-1]["mounts"][0]["ownership"], "pending")
        self.assertNotIn("identity", snapshots[-1]["mounts"][0])

    def test_preparation_rollback_revalidates_identity_before_unmount(self):
        path = "/mnt/colibri-test"
        expected = {
            "mount_id": 61,
            "device": "0:61",
            "filesystem": "tmpfs",
            "source": "tmpfs",
        }
        replacement = dict(expected, mount_id=62, device="0:62")
        manifest = {
            "state": "error",
            "plan": {
                "hardware": hardware_fixture(),
                "mounts": [{"path": path, "node": None}],
            },
            "mounts": [
                {
                    "path": path,
                    "node": None,
                    "ownership": "identified",
                    "identity": expected,
                }
            ],
        }
        unmount = mock.Mock()

        failures, retained, released = (
            lifecycle_support._rollback_preparation_mounts(
                manifest,
                mount_at=mock.Mock(side_effect=[expected, replacement]),
                mount_table=mock.Mock(return_value=[]),
                path_is_below=ramdisk._path_is_below,
                busy_mount_references=mock.Mock(return_value=[]),
                umount_path=unmount,
                validate_mount=mock.Mock(return_value=expected),
            )
        )

        self.assertTrue(failures)
        self.assertEqual(retained, {path})
        self.assertEqual(released, set())
        unmount.assert_not_called()

    def test_incomplete_busy_scan_withholds_every_preparation_unmount(self):
        paths = [
            "/mnt/colibri-test/node0",
            "/mnt/colibri-test/node1",
        ]
        identities = {
            path: {
                "mount_id": 71 + index,
                "device": "0:%d" % (71 + index),
                "filesystem": "tmpfs",
                "source": "tmpfs",
            }
            for index, path in enumerate(paths)
        }
        hardware = hardware_fixture(nodes=2)
        manifest = {
            "state": "error",
            "plan": {
                "hardware": hardware,
                "mounts": [
                    {"path": path, "node": index}
                    for index, path in enumerate(paths)
                ],
            },
            "mounts": [
                {
                    "path": path,
                    "node": index,
                    "ownership": "identified",
                    "identity": identities[path],
                }
                for index, path in enumerate(paths)
            ],
        }
        scans = []

        def busy_references(path, *, hardware=None):
            scans.append((path, hardware))
            if path == paths[0]:
                raise ramdisk.RamdiskError(
                    "managed cleanup requires complete /proc visibility"
                )
            return []

        unmount = mock.Mock()
        failures, retained, released = (
            lifecycle_support._rollback_preparation_mounts(
                manifest,
                mount_at=lambda path: identities[path],
                mount_table=mock.Mock(return_value=[]),
                path_is_below=ramdisk._path_is_below,
                busy_mount_references=busy_references,
                umount_path=unmount,
                validate_mount=(
                    lambda record, ignored_plan: record["identity"]
                ),
            )
        )

        self.assertTrue(
            any("complete /proc visibility" in item for item in failures)
        )
        self.assertEqual(retained, set(paths))
        self.assertEqual(released, set())
        self.assertTrue(all(item[1] is hardware for item in scans))
        unmount.assert_not_called()

    def test_preparation_rollback_requires_post_unmount_absence(self):
        path = "/mnt/colibri-test"
        expected = {
            "mount_id": 63,
            "device": "0:63",
            "filesystem": "tmpfs",
            "source": "tmpfs",
        }
        replacement = dict(expected, mount_id=64, device="0:64")

        for after in (expected, replacement):
            with self.subTest(after_mount_id=after["mount_id"]):
                record = {
                    "path": path,
                    "node": None,
                    "ownership": "identified",
                    "identity": expected,
                }
                manifest = {
                    "state": "error",
                    "plan": {
                        "hardware": hardware_fixture(),
                        "mounts": [{"path": path, "node": None}],
                    },
                    "mounts": [record],
                }
                unmount = mock.Mock()
                failures, retained, released = (
                    lifecycle_support._rollback_preparation_mounts(
                        manifest,
                        mount_at=mock.Mock(
                            side_effect=[
                                expected,
                                expected,
                                expected,
                                expected,
                                after,
                            ]
                        ),
                        mount_table=mock.Mock(return_value=[]),
                        path_is_below=ramdisk._path_is_below,
                        busy_mount_references=mock.Mock(return_value=[]),
                        umount_path=unmount,
                        validate_mount=mock.Mock(return_value=expected),
                    )
                )

                self.assertTrue(failures)
                self.assertEqual(retained, {path})
                self.assertEqual(released, set())
                unmount.assert_called_once()
                self.assertIn("remains or was replaced", failures[0])

    def test_preparation_rollback_rechecks_identity_after_busy_scan(self):
        path = "/mnt/colibri-test"
        expected = {
            "mount_id": 65,
            "device": "0:65",
            "filesystem": "tmpfs",
            "source": "tmpfs",
        }
        replacement = dict(expected, mount_id=66, device="0:66")
        current = {"identity": expected}
        busy_calls = {"count": 0}

        def busy(_path, hardware=None):
            busy_calls["count"] += 1
            if busy_calls["count"] == 3:
                current["identity"] = replacement
            return []

        record = {
            "path": path,
            "node": None,
            "ownership": "identified",
            "identity": expected,
        }
        unmount = mock.Mock()
        failures, retained, released = (
            lifecycle_support._rollback_preparation_mounts(
                {
                    "state": "error",
                    "plan": {
                        "hardware": hardware_fixture(),
                        "mounts": [{"path": path, "node": None}],
                    },
                    "mounts": [record],
                },
                mount_at=lambda ignored: current["identity"],
                mount_table=lambda: [],
                path_is_below=ramdisk._path_is_below,
                busy_mount_references=busy,
                umount_path=unmount,
                validate_mount=lambda ignored, plan: current["identity"],
            )
        )

        self.assertTrue(failures)
        self.assertEqual(retained, {path})
        self.assertEqual(released, set())
        self.assertIn("after busy scan", failures[0])
        unmount.assert_not_called()

    def test_copy_uses_atomic_publish_validates_header_and_removes_source_cache(self):
        with ModelFixture() as fixture, canonical_temporary_directory() as destination:
            source = fixture.root / "model-00001-of-00002.safetensors"
            target = Path(destination) / source.name
            ramdisk._copy_one(
                str(source), str(target), source.stat().st_size, 0, available=lambda: ramdisk.GIB
            )
            self.assertEqual(target.stat().st_size, source.stat().st_size)
            self.assertEqual(target.stat().st_mode & 0o222, 0)
            self.assertFalse(any(".coli-copy-" in item.name for item in Path(destination).iterdir()))

    def test_copy_stream_uses_binary_descriptors_and_preserves_ctrl_z(self):
        binary_flag = 1 << 28
        payload = b"safetensors-prefix\x1asafetensors-suffix"
        opened = []
        real_open = mounts_support.os.open

        def recording_open(path, flags, mode=0o777):
            opened.append((path, flags))
            return real_open(path, flags & ~binary_flag, mode)

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.bin"
            destination = Path(directory) / "destination.bin"
            source.write_bytes(payload)
            with mock.patch.object(
                mounts_support.os,
                "O_BINARY",
                binary_flag,
                create=True,
            ), mock.patch.object(
                mounts_support.os,
                "open",
                side_effect=recording_open,
            ):
                mounts_support._copy_stream(
                    str(source),
                    str(destination),
                    len(payload),
                )

            self.assertEqual(destination.read_bytes(), payload)

        self.assertEqual(len(opened), 2)
        self.assertTrue(opened[0][1] & binary_flag, "source descriptor must be binary")
        self.assertTrue(opened[1][1] & binary_flag, "destination descriptor must be binary")

    def test_mount_validation_rejects_foreign_filesystem(self):
        with ModelFixture() as fixture:
            plan = ramdisk.build_plan(plan_args(fixture.root), hardware=hardware_fixture())
        mount = plan["mounts"][0]
        foreign = {
            "mount_id": 3,
            "device": "8:1",
            "filesystem": "ext4",
            "source": "/dev/sda1",
            "options": ["rw", "noatime", "nodev", "nosuid", "noexec"],
            "super_options": [],
        }
        with mock.patch.object(ramdisk, "_mount_at", return_value=foreign):
            with self.assertRaisesRegex(ramdisk.RamdiskError, "foreign"):
                ramdisk._validate_mount(mount, plan)

    def test_mount_validation_requires_managed_thp_numa_and_safety_options(self):
        with ModelFixture() as fixture:
            plan = ramdisk.build_plan(plan_args(fixture.root), hardware=hardware_fixture(nodes=2))
        mount = plan["mounts"][0]
        actual = {
            "mount_id": 9,
            "device": "0:42",
            "filesystem": "tmpfs",
            "source": "tmpfs",
            "options": ["rw", "noatime", "nodev", "nosuid", "noexec"],
            "super_options": [
                "mode=700",
                "noswap",
                "huge=within_size",
                "mpol=interleave=static:0-1",
            ],
        }
        with mock.patch.object(ramdisk, "_mount_at", return_value=actual):
            self.assertEqual(ramdisk._validate_mount(mount, plan)["mount_id"], 9)
        actual["super_options"].remove("mpol=interleave=static:0-1")
        with mock.patch.object(ramdisk, "_mount_at", return_value=actual):
            with self.assertRaisesRegex(ramdisk.RamdiskError, "NUMA policy"):
                ramdisk._validate_mount(mount, plan)

    def test_single_selected_node_rejects_sampled_pages_on_another_host_node(self):
        with ModelFixture() as fixture, canonical_temporary_directory() as destination:
            plan = ramdisk.build_plan(
                plan_args(fixture.root, memory_nodes="0"),
                hardware=hardware_fixture(nodes=2),
            )
            self._stage_fixture_namespace(fixture, destination)
            mount = {"path": destination, "node": None}
            with mock.patch.object(
                ramdisk,
                "_sample_numa_allocation",
                return_value={"1": 128},
            ):
                with self.assertRaisesRegex(
                    ramdisk.RamdiskError, "escaped the reviewed NUMA"
                ):
                    ramdisk._validate_namespace(plan, mount)

    def test_interleaved_namespace_accepts_balanced_sample(self):
        with ModelFixture() as fixture, canonical_temporary_directory() as destination:
            plan = ramdisk.build_plan(
                plan_args(fixture.root),
                hardware=hardware_fixture(nodes=2),
            )
            self._stage_fixture_namespace(fixture, destination)
            mount = {"path": destination, "node": None}
            with mock.patch.object(
                ramdisk,
                "_sample_numa_allocation",
                return_value={"0": 50, "1": 50},
            ):
                self.assertEqual(
                    ramdisk._validate_namespace(plan, mount),
                    {"0": 100, "1": 100},
                )

    def test_interleaved_namespace_reports_imbalanced_sample(self):
        with ModelFixture() as fixture, canonical_temporary_directory() as destination:
            plan = ramdisk.build_plan(
                plan_args(fixture.root),
                hardware=hardware_fixture(nodes=2),
            )
            self._stage_fixture_namespace(fixture, destination)
            mount = {"path": destination, "node": None}
            with mock.patch.object(
                ramdisk,
                "_sample_numa_allocation",
                return_value={"0": 60, "1": 40},
            ):
                with self.assertRaises(ramdisk.RamdiskError) as raised:
                    ramdisk._validate_namespace(plan, mount)
            message = str(raised.exception)
            self.assertIn("0=120, 1=80", message)
            self.assertIn("20.0%", message)
            self.assertIn("exceeds 15%", message)
            self.assertNotIn("%%", message)

    def test_node_local_namespace_uses_single_percent_sign(self):
        with ModelFixture() as fixture, canonical_temporary_directory() as destination:
            plan = ramdisk.build_plan(
                plan_args(fixture.root, topology="per-node"),
                hardware=hardware_fixture(nodes=2),
            )
            self._stage_fixture_namespace(fixture, destination)
            mount = {"path": destination, "node": 0}
            with mock.patch.object(
                ramdisk,
                "_sample_numa_allocation",
                return_value={"0": 90, "1": 10},
            ):
                with self.assertRaises(ramdisk.RamdiskError) as raised:
                    ramdisk._validate_namespace(plan, mount)
            self.assertIn("95% local allocation", str(raised.exception))
            self.assertNotIn("%%", str(raised.exception))

    def test_mount_lookup_rejects_stacked_exact_paths(self):
        mounts = [
            {"mount_id": 4, "path": "/mnt/colibri-test"},
            {"mount_id": 9, "path": "/mnt/colibri-test"},
        ]
        with mock.patch.object(ramdisk, "_mount_table", return_value=mounts):
            with self.assertRaisesRegex(ramdisk.RamdiskError, "stacked mounts"):
                ramdisk._mount_at("/mnt/colibri-test")

    def test_filesystem_lookup_rejects_stacked_longest_mountpoint(self):
        mounts = [
            {
                "mount_id": 41,
                "parent_id": 1,
                "path": "/durable",
                "filesystem": "xfs",
            },
            {
                "mount_id": 42,
                "parent_id": 41,
                "path": "/durable",
                "filesystem": "tmpfs",
            },
        ]
        with mock.patch.object(ramdisk, "_mount_table", return_value=mounts):
            with self.assertRaisesRegex(
                ramdisk.RamdiskError,
                "ambiguous stacked mounts",
            ):
                ramdisk._filesystem_for_path("/durable/manifest.json")
