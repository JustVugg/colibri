"""RAM-disk mount, copy, and staged-namespace tests."""

if __package__:
    from .ramdisk_test_support import *  # noqa: F401,F403
else:
    from ramdisk_test_support import *  # noqa: F401,F403


class MountAndCopyTest(unittest.TestCase):
    def test_mountinfo_preserves_noncontiguous_mpol_nodemask(self):
        line = (
            "36 25 0:32 / /mnt/colibri-ram rw,noatime - tmpfs tmpfs "
            "rw,noswap,nodev,nosuid,noexec,mode=700,huge=within_size,"
            "mpol=interleave:0-1,3\n"
        )
        with tempfile.NamedTemporaryFile("w", encoding="utf-8") as stream:
            stream.write(line)
            stream.flush()
            parsed = ramdisk._mount_table(stream.name)
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

    def test_interrupted_successful_mount_is_identified_and_rolled_back_locally(self):
        with ModelFixture() as fixture:
            plan = ramdisk.build_plan(
                plan_args(fixture.root),
                hardware=hardware_fixture(),
            )
        mount = plan["mounts"][0]
        actual = {
            "mount_id": 44,
            "device": "0:44",
            "filesystem": "tmpfs",
            "source": "tmpfs",
        }
        interrupted = ramdisk._TuiTerminationSignal(signal.SIGTERM)
        with mock.patch.object(
            ramdisk,
            "_trusted_system_binary",
            return_value="/bin/mount",
        ), mock.patch.object(
            ramdisk,
            "_privileged",
            side_effect=lambda command, hardware: command,
        ), mock.patch.object(
            ramdisk,
            "_run",
            side_effect=interrupted,
        ), mock.patch.object(
            ramdisk,
            "_mount_at",
            return_value=actual,
        ), mock.patch.object(
            ramdisk,
            "_validate_mount",
            return_value=actual,
        ) as validate, mock.patch.object(
            ramdisk,
            "_umount_path",
        ) as unmount:
            with self.assertRaises(ramdisk._TuiTerminationSignal):
                ramdisk._mount_tmpfs(plan, mount)

        attempted = validate.call_args.args[0]
        self.assertEqual(attempted["effective_thp"], "within_size")
        self.assertTrue(attempted["effective_noswap"])
        unmount.assert_called_once_with(mount["path"], plan["hardware"])

    def test_prepare_immediately_rolls_back_identityless_successful_mount(self):
        with ModelFixture() as fixture, mock.patch.object(
            ramdisk, "_filesystem_for_path", return_value="ext4"
        ):
            plan = ramdisk.build_plan(
                plan_args(fixture.root), hardware=hardware_fixture()
            )
        with mock.patch.object(ramdisk, "_load_manifest", return_value=None), mock.patch.object(
            ramdisk, "build_plan", return_value=plan
        ), mock.patch.object(ramdisk, "_save_manifest"), mock.patch.object(
            ramdisk, "_mount_at", side_effect=[None, None]
        ), mock.patch.object(ramdisk, "_mount_tmpfs"), mock.patch.object(
            ramdisk, "_umount_path"
        ) as unmount:
            with self.assertRaisesRegex(ramdisk.RamdiskError, "rolled it back"):
                ramdisk.prepare.__wrapped__(
                    plan_args(fixture.root, yes=True), display_plan=False
                )
        unmount.assert_called_once_with(plan["mounts"][0]["path"], plan["hardware"])

    def test_prepare_cleanup_runs_even_when_error_manifest_cannot_be_saved(self):
        with ModelFixture() as fixture, mock.patch.object(
            ramdisk, "_filesystem_for_path", return_value="ext4"
        ):
            plan = ramdisk.build_plan(
                plan_args(fixture.root), hardware=hardware_fixture()
            )
        actual = {
            "mount_id": 9,
            "device": "0:42",
            "filesystem": "tmpfs",
            "source": "tmpfs",
        }
        with mock.patch.object(ramdisk, "_load_manifest", return_value=None), mock.patch.object(
            ramdisk, "build_plan", return_value=plan
        ), mock.patch.object(
            ramdisk, "_save_manifest", side_effect=[None, OSError("state full"), OSError("state full"), OSError("state full")]
        ), mock.patch.object(
            ramdisk, "_mount_at", side_effect=[None, actual, actual]
        ), mock.patch.object(ramdisk, "_mount_tmpfs"), mock.patch.object(
            ramdisk, "_umount_path"
        ) as unmount:
            with self.assertRaisesRegex(ramdisk.RamdiskError, "rollback/reporting errors"):
                ramdisk.prepare.__wrapped__(
                    plan_args(fixture.root, yes=True), display_plan=False
                )
        unmount.assert_called_once_with(plan["mounts"][0]["path"], plan["hardware"])

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
            for source in fixture.root.glob("*.safetensors"):
                target = Path(destination) / source.name
                shutil.copy2(source, target)
                target.chmod(0o400)
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

    def test_mount_lookup_rejects_stacked_exact_paths(self):
        mounts = [
            {"mount_id": 4, "path": "/mnt/colibri-test"},
            {"mount_id": 9, "path": "/mnt/colibri-test"},
        ]
        with mock.patch.object(ramdisk, "_mount_table", return_value=mounts):
            with self.assertRaisesRegex(ramdisk.RamdiskError, "stacked mounts"):
                ramdisk._mount_at("/mnt/colibri-test")

