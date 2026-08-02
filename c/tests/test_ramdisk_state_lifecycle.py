"""RAM-disk durable state, safety, and lifecycle tests."""

import copy

if __package__:
    from .ramdisk_test_support import *  # noqa: F401,F403
else:
    from ramdisk_test_support import *  # noqa: F401,F403

from ramdisk_support import lifecycle as lifecycle_support
from ramdisk_support import linux_ops
from ramdisk_support import processes as process_support
from ramdisk_support import state as state_support


_REAL_BOUND_PARENT_DESCRIPTOR = state_support._bound_parent_descriptor
_REAL_FSYNC_DIRECTORY = state_support._fsync_directory


@contextlib.contextmanager
def _portable_descriptor_seam():
    """Exercise descriptor-gated state logic without weakening production."""

    def allow_portable_binding(*args, **kwargs):
        kwargs["require_native"] = False
        return _REAL_BOUND_PARENT_DESCRIPTOR(*args, **kwargs)

    with mock.patch.object(
        state_support,
        "_bound_parent_descriptor",
        new=allow_portable_binding,
    ), mock.patch.object(
        state_support,
        "_fsync_bound_directory",
        new=lambda descriptor: None,
    ), mock.patch.object(
        state_support,
        "_fsync_directory",
        new=lambda path: None,
    ):
        yield


class StateAndSafetyTest(unittest.TestCase):
    FINGERPRINT = "sha256:" + ("a" * 64)
    GLM_ENGINE_ID = 3815245270
    USAGE_HEADER = "-1 1 2\n-2 1 %d\n" % GLM_ENGINE_ID

    def setUp(self):
        self.descriptor_seam = contextlib.ExitStack()
        self.addCleanup(self.descriptor_seam.close)
        if not state_support._supports_native_dirfd():
            self.descriptor_seam.enter_context(_portable_descriptor_seam())
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
        self.descriptor_seam.close()
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

    def test_recovery_uses_one_canonical_counts_and_markers_snapshot(self):
        model_dir = os.path.join(self.root, "model")
        state_dir = os.path.join(self.root, "node-state")
        os.makedirs(model_dir)
        os.makedirs(state_dir)
        canonical = os.path.join(model_dir, ".coli_usage")
        merge_id = "4" * 32
        ramdisk._usage_write(canonical, {"0:1": 10})
        ramdisk._atomic_json(
            os.path.join(state_dir, ".coli_usage.delta.json"),
            {"version": 1, "id": merge_id, "delta": {"0:1": 2}},
        )
        real_snapshot = state_support._usage_snapshot_from_bound
        canonical_snapshots = []

        def observe_snapshot(bound, name, path, **kwargs):
            snapshot = real_snapshot(bound, name, path, **kwargs)
            if kwargs.get("source") == "canonical usage target":
                canonical_snapshots.append(snapshot["text"])
            return snapshot

        with mock.patch.object(
            state_support,
            "_usage_snapshot_from_bound",
            side_effect=observe_snapshot,
        ):
            ramdisk._recover_delta(state_dir, canonical)

        self.assertEqual(len(canonical_snapshots), 1)
        self.assertEqual(ramdisk._usage_read(canonical)["0:1"], 12)
        self.assertEqual(ramdisk._usage_merge_ids(canonical), {merge_id})

    def test_usage_delta_recovery_rejects_nonpositive_or_coerced_counts(self):
        model_dir = os.path.join(self.root, "model")
        canonical = os.path.join(model_dir, ".coli_usage")
        state_dir = os.path.join(self.root, "node-state")
        os.makedirs(model_dir)
        os.makedirs(state_dir)
        ramdisk._usage_write(canonical, {"0:1": 10})
        delta_path = os.path.join(state_dir, ".coli_usage.delta.json")

        invalid_deltas = (
            {"0:1": 0},
            {"0:1": -3},
            {"0:1": True},
            {"0:1": 2.5},
            {"0:1": "3"},
            {"invalid": 3},
        )
        for index, delta in enumerate(invalid_deltas, 1):
            with self.subTest(delta=delta):
                ramdisk._atomic_json(
                    delta_path,
                    {
                        "version": 1,
                        "id": ("%x" % index) * 32,
                        "delta": delta,
                    },
                )
                journal_before = Path(delta_path).read_bytes()
                canonical_before = Path(canonical).read_bytes()

                with self.assertRaisesRegex(
                    ramdisk.RamdiskError,
                    "usage delta journal has invalid counts",
                ):
                    ramdisk._recover_delta(state_dir, canonical)

                self.assertEqual(Path(canonical).read_bytes(), canonical_before)
                self.assertEqual(Path(delta_path).read_bytes(), journal_before)

    def test_present_null_usage_journal_is_malformed_and_retained(self):
        model_dir = os.path.join(self.root, "model")
        state_dir = os.path.join(self.root, "node-state")
        os.makedirs(model_dir)
        os.makedirs(state_dir)
        canonical = os.path.join(model_dir, ".coli_usage")
        delta_path = os.path.join(state_dir, ".coli_usage.delta.json")
        ramdisk._usage_write(canonical, {"0:1": 10})
        Path(delta_path).write_text("null\n", encoding="utf-8")
        canonical_before = Path(canonical).read_bytes()

        with self.assertRaisesRegex(
            ramdisk.RamdiskError,
            "must contain a JSON object",
        ):
            ramdisk._recover_delta(state_dir, canonical)

        self.assertEqual(Path(canonical).read_bytes(), canonical_before)
        self.assertEqual(Path(delta_path).read_text(encoding="utf-8"), "null\n")

    def test_absent_usage_journal_retry_reproves_parent_directory(self):
        model_dir = os.path.join(self.root, "model")
        state_dir = os.path.join(self.root, "node-state")
        os.makedirs(model_dir)
        os.makedirs(state_dir)
        canonical = os.path.join(model_dir, ".coli_usage")
        ramdisk._usage_write(canonical, {"0:1": 10})

        with mock.patch.object(
            state_support,
            "_fsync_bound_directory",
        ) as sync_directory, mock.patch.object(
            state_support,
            "_fsync_directory",
            new=sync_directory,
        ):
            ramdisk._recover_delta(state_dir, canonical)

        sync_directory.assert_called_once()

    def test_record_merge_refuses_mismatched_journal_transaction(self):
        model_dir = os.path.join(self.root, "model")
        canonical = os.path.join(model_dir, ".coli_usage")
        state_dir = os.path.join(self.root, "node-state")
        os.makedirs(model_dir)
        os.makedirs(state_dir)
        ramdisk._usage_write(canonical, {"0:1": 10})
        ramdisk._usage_write(
            os.path.join(state_dir, ".coli_usage"),
            {"0:1": 12},
        )
        delta_path = os.path.join(state_dir, ".coli_usage.delta.json")
        ramdisk._atomic_json(
            delta_path,
            {
                "version": 1,
                "id": "b" * 32,
                "delta": {"0:1": 7},
            },
        )
        canonical_before = Path(canonical).read_bytes()
        journal_before = Path(delta_path).read_bytes()

        with self.assertRaisesRegex(
            ramdisk.RamdiskError,
            "journal transaction.*managed record",
        ):
            ramdisk._merge_usage(
                {
                    "state_dir": state_dir,
                    "usage_baseline": {"0:1": 10},
                    "usage_merge_id": "a" * 32,
                },
                canonical,
            )

        self.assertEqual(Path(canonical).read_bytes(), canonical_before)
        self.assertEqual(Path(delta_path).read_bytes(), journal_before)

    def test_recover_delta_facade_forwards_expected_transaction(self):
        plan = {"identity": "test-plan"}
        merge_id = "a" * 32
        with mock.patch.object(
            ramdisk,
            "_state_recover_delta",
            return_value=merge_id,
        ) as recover:
            result = ramdisk._recover_delta(
                "/durable/state",
                "/model/.coli_usage",
                plan=plan,
                expected_merge_id=merge_id,
            )

        self.assertEqual(result, merge_id)
        recover.assert_called_once_with(
            "/durable/state",
            "/model/.coli_usage",
            plan=plan,
            expected_merge_id=merge_id,
            filesystem_for_path=ramdisk._filesystem_for_path,
            source_still_matches=ramdisk._source_still_matches,
        )

    def test_matching_record_journal_and_standalone_legacy_journal_recover(self):
        model_dir = os.path.join(self.root, "model")
        canonical = os.path.join(model_dir, ".coli_usage")
        state_dir = os.path.join(self.root, "node-state")
        standalone_dir = os.path.join(self.root, "standalone-state")
        os.makedirs(model_dir)
        os.makedirs(state_dir)
        os.makedirs(standalone_dir)
        ramdisk._usage_write(canonical, {"0:1": 10})
        ramdisk._usage_write(
            os.path.join(state_dir, ".coli_usage"),
            {"0:1": 12},
        )
        merge_id = "c" * 32
        ramdisk._atomic_json(
            os.path.join(state_dir, ".coli_usage.delta.json"),
            {"version": 1, "id": merge_id, "delta": {"0:1": 2}},
        )

        ramdisk._merge_usage(
            {
                "state_dir": state_dir,
                "usage_baseline": {"0:1": 10},
                "usage_merge_id": merge_id,
            },
            canonical,
        )

        self.assertEqual(ramdisk._usage_read(canonical)["0:1"], 12)
        standalone_id = "d" * 32
        ramdisk._atomic_json(
            os.path.join(standalone_dir, ".coli_usage.delta.json"),
            {
                "version": 1,
                "id": standalone_id,
                "delta": {"0:1": 1},
            },
        )
        ramdisk._recover_delta(standalone_dir, canonical)
        self.assertEqual(ramdisk._usage_read(canonical)["0:1"], 13)
        self.assertEqual(
            ramdisk._usage_merge_ids(canonical),
            {merge_id, standalone_id},
        )

    def test_legacy_record_adopts_existing_journal_transaction_idempotently(self):
        model_dir = os.path.join(self.root, "model")
        canonical = os.path.join(model_dir, ".coli_usage")
        state_dir = os.path.join(self.root, "node-state")
        os.makedirs(model_dir)
        os.makedirs(state_dir)
        ramdisk._usage_write(canonical, {"0:1": 10})
        ramdisk._usage_write(
            os.path.join(state_dir, ".coli_usage"),
            {"0:1": 12},
        )
        merge_id = "9" * 32
        delta_path = os.path.join(state_dir, ".coli_usage.delta.json")
        journal = {
            "version": 1,
            "id": merge_id,
            "delta": {"0:1": 2},
        }
        ramdisk._atomic_json(delta_path, journal)
        record = {
            "state_dir": state_dir,
            "usage_baseline": {"0:1": 10},
        }

        ramdisk._merge_usage(record, canonical)

        self.assertEqual(record["usage_merge_id"], merge_id)
        self.assertEqual(ramdisk._usage_read(canonical)["0:1"], 12)
        self.assertFalse(os.path.exists(delta_path))

        # A stale replay of the adopted legacy journal must remain idempotent.
        ramdisk._atomic_json(delta_path, journal)
        ramdisk._merge_usage(record, canonical)
        self.assertEqual(ramdisk._usage_read(canonical)["0:1"], 12)
        self.assertFalse(os.path.exists(delta_path))

    def test_usage_history_read_is_optional_only_when_absent(self):
        missing = os.path.join(self.root, "missing-usage")
        self.assertEqual(state_support._usage_read(missing), {})

        denied = os.path.join(self.root, "denied-usage")
        Path(denied).write_text("0 1 7\n", encoding="utf-8")
        real_open = state_support.os.open

        def deny_usage_open(path, flags, *args, **kwargs):
            if path in (denied, os.path.basename(denied)):
                raise PermissionError("permission denied")
            return real_open(path, flags, *args, **kwargs)

        with mock.patch.object(
            state_support.os,
            "open",
            side_effect=deny_usage_open,
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

    def test_managed_usage_merge_requires_regular_nonsymlink_state_file(self):
        model_dir = os.path.join(self.root, "model")
        canonical = os.path.join(model_dir, ".coli_usage")
        os.makedirs(model_dir)
        ramdisk._usage_write(canonical, {"0:1": 10})

        for kind in ("missing", "directory", "symlink"):
            with self.subTest(kind=kind):
                state_dir = os.path.join(self.root, "state-" + kind)
                os.makedirs(state_dir)
                state_usage = os.path.join(state_dir, ".coli_usage")
                if kind == "directory":
                    os.mkdir(state_usage)
                elif kind == "symlink":
                    target = os.path.join(self.root, "usage-target")
                    Path(target).write_text("0 1 12\n", encoding="utf-8")
                    os.symlink(target, state_usage)
                canonical_before = Path(canonical).read_bytes()

                with self.assertRaisesRegex(
                    ramdisk.RamdiskError,
                    "managed usage history.*regular non-symlink file",
                ):
                    ramdisk._merge_usage(
                        {
                            "state_dir": state_dir,
                            "usage_baseline": {"0:1": 10},
                            "usage_merge_id": "e" * 32,
                        },
                        canonical,
                    )

                self.assertEqual(Path(canonical).read_bytes(), canonical_before)

    @requires_native_dirfd
    def test_managed_usage_merge_rejects_symlink_swap_during_verified_open(self):
        model_dir = os.path.join(self.root, "model")
        canonical = os.path.join(model_dir, ".coli_usage")
        state_dir = os.path.join(self.root, "state-native")
        state_usage = os.path.join(state_dir, ".coli_usage")
        attacker = os.path.join(self.root, "attacker-native")
        os.makedirs(model_dir)
        os.makedirs(state_dir)
        ramdisk._usage_write(canonical, {"0:1": 10})
        Path(state_usage).write_text("0 1 12\n", encoding="utf-8")
        Path(attacker).write_text("0 1 99\n", encoding="utf-8")
        real_open = state_support.os.open
        swapped = False

        def swap_before_open(path, flags, *args, **kwargs):
            nonlocal swapped
            if path in (state_usage, ".coli_usage") and not swapped:
                swapped = True
                os.unlink(state_usage)
                os.symlink(attacker, state_usage)
            return real_open(path, flags, *args, **kwargs)

        with mock.patch.object(
            state_support.os,
            "open",
            side_effect=swap_before_open,
        ):
            with self.assertRaisesRegex(
                ramdisk.RamdiskError,
                "managed usage history.*regular non-symlink|"
                "managed usage history changed|"
                "cannot read managed usage history",
            ):
                ramdisk._merge_usage(
                    {
                        "state_dir": state_dir,
                        "usage_baseline": {"0:1": 10},
                        "usage_merge_id": "8" * 32,
                    },
                    canonical,
                )

        self.assertTrue(swapped)
        self.assertEqual(ramdisk._usage_read(canonical)["0:1"], 10)

    def test_managed_usage_merge_fails_closed_without_native_dirfd(self):
        model_dir = os.path.join(self.root, "model")
        canonical = os.path.join(model_dir, ".coli_usage")
        state_dir = os.path.join(self.root, "state-portable")
        os.makedirs(model_dir)
        os.makedirs(state_dir)
        ramdisk._usage_write(canonical, {"0:1": 10})
        ramdisk._usage_write(
            os.path.join(state_dir, ".coli_usage"),
            {"0:1": 12},
        )
        canonical_before = Path(canonical).read_bytes()

        with mock.patch.object(
            state_support,
            "_bound_parent_descriptor",
            new=_REAL_BOUND_PARENT_DESCRIPTOR,
        ), mock.patch.object(
            state_support,
            "_supports_native_dirfd",
            return_value=False,
        ), mock.patch.object(
            state_support.os,
            "open",
            wraps=state_support.os.open,
        ) as open_file:
            with self.assertRaisesRegex(
                ramdisk.RamdiskError,
                "managed state requires descriptor-relative filesystem operations",
            ):
                ramdisk._merge_usage(
                    {
                        "state_dir": state_dir,
                        "usage_baseline": {"0:1": 10},
                        "usage_merge_id": "9" * 32,
                    },
                    canonical,
                )

        open_file.assert_not_called()
        self.assertEqual(Path(canonical).read_bytes(), canonical_before)

    def test_managed_usage_read_binds_parent_identity(self):
        model_dir = os.path.join(self.root, "model")
        canonical = os.path.join(model_dir, ".coli_usage")
        state_dir = os.path.join(self.root, "node-state")
        original_dir = os.path.join(self.root, "original-state")
        replacement_dir = os.path.join(self.root, "replacement-state")
        os.makedirs(model_dir)
        os.makedirs(state_dir)
        os.makedirs(replacement_dir)
        ramdisk._usage_write(canonical, {"0:1": 10})
        ramdisk._usage_write(
            os.path.join(state_dir, ".coli_usage"),
            {"0:1": 12},
        )
        ramdisk._usage_write(
            os.path.join(replacement_dir, ".coli_usage"),
            {"0:1": 99},
        )
        state_usage = os.path.join(state_dir, ".coli_usage")
        real_open = state_support.os.open
        swapped = False

        def swap_parent_before_open(path, flags, *args, **kwargs):
            nonlocal swapped
            if path in (state_dir, state_usage) and not swapped:
                swapped = True
                os.rename(state_dir, original_dir)
                os.rename(replacement_dir, state_dir)
            return real_open(path, flags, *args, **kwargs)

        with mock.patch.object(
            state_support.os,
            "open",
            side_effect=swap_parent_before_open,
        ):
            with self.assertRaisesRegex(
                ramdisk.RamdiskError,
                "managed state|parent.*changed|managed usage history",
            ):
                ramdisk._merge_usage(
                    {
                        "state_dir": state_dir,
                        "usage_baseline": {"0:1": 10},
                        "usage_merge_id": "7" * 32,
                    },
                    canonical,
                )

        self.assertTrue(swapped)
        self.assertEqual(ramdisk._usage_read(canonical)["0:1"], 10)

    def test_managed_usage_seed_write_rejects_symlink_target(self):
        state_dir = os.path.join(self.root, "node-state")
        os.makedirs(state_dir)
        state_usage = os.path.join(state_dir, ".coli_usage")
        attacker = os.path.join(self.root, "attacker-usage")
        Path(attacker).write_text("0 1 99\n", encoding="utf-8")
        os.symlink(attacker, state_usage)

        with self.assertRaisesRegex(
            ramdisk.RamdiskError,
            "managed usage history.*regular non-symlink",
        ):
            state_support._managed_usage_write(
                state_usage,
                {"0:1": 10},
                filesystem_for_path=lambda path: "ext4",
            )

        self.assertTrue(os.path.islink(state_usage))
        self.assertIn("0 1 99", Path(attacker).read_text(encoding="utf-8"))

    @requires_native_dirfd
    def test_managed_usage_seed_write_binds_parent_identity(self):
        state_dir = os.path.join(self.root, "node-state")
        original_dir = os.path.join(self.root, "original-state")
        replacement_dir = os.path.join(self.root, "replacement-state")
        os.makedirs(state_dir)
        os.makedirs(replacement_dir)
        state_usage = os.path.join(state_dir, ".coli_usage")
        real_open = state_support.os.open
        swapped = False

        def swap_parent_before_open(path, flags, *args, **kwargs):
            nonlocal swapped
            if path == state_dir and not swapped:
                swapped = True
                os.rename(state_dir, original_dir)
                os.rename(replacement_dir, state_dir)
            return real_open(path, flags, *args, **kwargs)

        with mock.patch.object(
            state_support.os,
            "open",
            side_effect=swap_parent_before_open,
        ):
            with self.assertRaisesRegex(
                ramdisk.RamdiskError,
                "managed state.*parent identity changed|"
                "managed state parent identity changed",
            ):
                state_support._managed_usage_write(
                    state_usage,
                    {"0:1": 10},
                    filesystem_for_path=lambda path: "ext4",
                )

        self.assertTrue(swapped)
        self.assertFalse(os.path.exists(os.path.join(state_dir, ".coli_usage")))
        self.assertFalse(os.path.exists(os.path.join(original_dir, ".coli_usage")))

    @requires_posix_fifo
    def test_managed_usage_swap_to_fifo_uses_nonblocking_open(self):
        model_dir = os.path.join(self.root, "model")
        canonical = os.path.join(model_dir, ".coli_usage")
        state_dir = os.path.join(self.root, "node-state")
        os.makedirs(model_dir)
        os.makedirs(state_dir)
        ramdisk._usage_write(canonical, {"0:1": 10})
        state_usage = os.path.join(state_dir, ".coli_usage")
        ramdisk._usage_write(state_usage, {"0:1": 12})
        real_open = state_support.os.open
        swapped = False
        opened_flags = []

        def swap_to_fifo_before_open(path, flags, *args, **kwargs):
            nonlocal swapped
            if path in (state_usage, ".coli_usage") and not swapped:
                swapped = True
                os.unlink(state_usage)
                os.mkfifo(state_usage)
                opened_flags.append(flags)
                if not flags & getattr(os, "O_NONBLOCK", 0):
                    raise AssertionError("FIFO open would block")
            return real_open(path, flags, *args, **kwargs)

        with mock.patch.object(
            state_support.os,
            "open",
            side_effect=swap_to_fifo_before_open,
        ):
            with self.assertRaisesRegex(
                ramdisk.RamdiskError,
                "regular non-symlink",
            ):
                ramdisk._merge_usage(
                    {
                        "state_dir": state_dir,
                        "usage_baseline": {"0:1": 10},
                        "usage_merge_id": "6" * 32,
                    },
                    canonical,
                )

        self.assertTrue(swapped)
        self.assertTrue(opened_flags[0] & os.O_NONBLOCK)
        self.assertEqual(ramdisk._usage_read(canonical)["0:1"], 10)

    def test_canonical_usage_symlink_cannot_authorize_marker_shortcut(self):
        model_dir = os.path.join(self.root, "model")
        state_dir = os.path.join(self.root, "node-state")
        os.makedirs(model_dir)
        os.makedirs(state_dir)
        canonical = os.path.join(model_dir, ".coli_usage")
        attacker = os.path.join(self.root, "attacker-usage")
        merge_id = "5" * 32
        Path(attacker).write_text(
            "0 1 99\n# coli-ramdisk-merge %s\n" % merge_id,
            encoding="utf-8",
        )
        os.symlink(attacker, canonical)
        ramdisk._usage_write(
            os.path.join(state_dir, ".coli_usage"),
            {"0:1": 12},
        )

        with self.assertRaisesRegex(
            ramdisk.RamdiskError,
            "canonical usage.*regular non-symlink|canonical usage target",
        ):
            ramdisk._merge_usage(
                {
                    "state_dir": state_dir,
                    "usage_baseline": {"0:1": 10},
                    "usage_merge_id": merge_id,
                },
                canonical,
            )

        self.assertTrue(os.path.islink(canonical))
        self.assertIn("0 1 99", Path(attacker).read_text(encoding="utf-8"))

    def test_managed_usage_merge_rejects_missing_or_regressed_positive_baseline(self):
        model_dir = os.path.join(self.root, "model")
        canonical = os.path.join(model_dir, ".coli_usage")
        state_dir = os.path.join(self.root, "node-state")
        os.makedirs(model_dir)
        os.makedirs(state_dir)
        ramdisk._usage_write(canonical, {"0:1": 10, "0:2": 4})
        state_usage = os.path.join(state_dir, ".coli_usage")

        for label, current, message in (
            ("missing", {"0:2": 4}, "missing positive baseline counter"),
            ("regressed", {"0:1": 9, "0:2": 4}, "regressed below baseline"),
        ):
            with self.subTest(label=label):
                ramdisk._usage_write(state_usage, current)
                canonical_before = Path(canonical).read_bytes()
                with self.assertRaisesRegex(ramdisk.RamdiskError, message):
                    ramdisk._merge_usage(
                        {
                            "state_dir": state_dir,
                            "usage_baseline": {"0:1": 10, "0:2": 4},
                            "usage_merge_id": ("1" if label == "missing" else "2") * 32,
                        },
                        canonical,
                    )
                self.assertEqual(Path(canonical).read_bytes(), canonical_before)

    def test_zero_baseline_omission_and_zero_byte_history_remain_valid(self):
        model_dir = os.path.join(self.root, "model")
        canonical = os.path.join(model_dir, ".coli_usage")
        state_dir = os.path.join(self.root, "node-state")
        os.makedirs(model_dir)
        os.makedirs(state_dir)
        Path(canonical).touch()
        state_usage = os.path.join(state_dir, ".coli_usage")
        Path(state_usage).touch()

        ramdisk._merge_usage(
            {
                "state_dir": state_dir,
                "usage_baseline": {"0:1": 0},
                "usage_merge_id": "3" * 32,
            },
            canonical,
        )

        self.assertEqual(Path(canonical).read_bytes(), b"")
        self.assertEqual(Path(state_usage).read_bytes(), b"")

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

    def test_multilayer_usage_header_survives_a_second_managed_merge(self):
        model_dir = os.path.join(self.root, "model")
        canonical = os.path.join(model_dir, ".coli_usage")
        state_dir = os.path.join(self.root, "node-state")
        os.makedirs(model_dir)
        os.makedirs(state_dir)
        header = state_support._usage_header_counts(
            {
                "n_layers": 40,
                "n_experts": 8,
                "format_version": 1,
                "engine_id": self.GLM_ENGINE_ID,
            }
        )
        first_baseline = dict(header)
        first_baseline["0:1"] = 10
        manifest = self.manifest(
            state="running",
            processes=[{"pid": 731}],
        )
        manifest["processes"][0]["usage_baseline"] = dict(first_baseline)
        ramdisk._atomic_json(ramdisk._manifest_path(), manifest)
        loaded = ramdisk._load_manifest(required=True)
        self.assertEqual(
            loaded["processes"][0]["usage_baseline"]["-1:40"],
            8,
        )

        ramdisk._usage_write(canonical, first_baseline)
        ramdisk._usage_write(
            os.path.join(state_dir, ".coli_usage"),
            dict(first_baseline, **{"0:1": 12}),
        )

        ramdisk._merge_usage(
            {
                "state_dir": state_dir,
                "usage_baseline": first_baseline,
                "usage_merge_id": "1" * 32,
            },
            canonical,
        )
        second_baseline = ramdisk._usage_read(canonical)
        self.assertEqual(second_baseline["-1:40"], 8)
        self.assertEqual(second_baseline["0:1"], 12)

        second_current = dict(second_baseline)
        second_current["0:1"] = 15
        ramdisk._usage_write(
            os.path.join(state_dir, ".coli_usage"),
            second_current,
        )
        ramdisk._merge_usage(
            {
                "state_dir": state_dir,
                "usage_baseline": second_baseline,
                "usage_merge_id": "2" * 32,
            },
            canonical,
        )

        final = ramdisk._usage_read(canonical)
        self.assertEqual(final["-1:40"], 8)
        self.assertEqual(final["-2:1"], self.GLM_ENGINE_ID)
        self.assertEqual(final["0:1"], 15)

    def test_manifest_rejects_nonpositive_or_malformed_usage_headers(self):
        invalid_baselines = (
            {"-1:0": 8, "-2:1": self.GLM_ENGINE_ID, "0:1": 10},
            {"-1:40": 8, "-2:0": self.GLM_ENGINE_ID, "0:1": 10},
            {"-1:40": 0, "-2:1": self.GLM_ENGINE_ID, "0:1": 10},
            {"-1:40": 8, "-2:1": 0, "0:1": 10},
            {"-1:forty": 8, "-2:1": self.GLM_ENGINE_ID, "0:1": 10},
        )
        for baseline in invalid_baselines:
            with self.subTest(baseline=baseline):
                manifest = self.manifest(
                    state="running",
                    processes=[{"pid": 731}],
                )
                manifest["processes"][0]["usage_baseline"] = baseline
                ramdisk._atomic_json(ramdisk._manifest_path(), manifest)
                with self.assertRaisesRegex(
                    ramdisk.RamdiskError,
                    "unsafe managed process record",
                ):
                    ramdisk._load_manifest(required=True)

    def test_usage_reader_rejects_zero_header_metadata(self):
        invalid_headers = (
            ("-1 0 8\n-2 1 %d\n" % self.GLM_ENGINE_ID, "dimensions"),
            ("-1 40 8\n-2 0 %d\n" % self.GLM_ENGINE_ID, "version"),
            ("-1 40 8\n-2 1 0\n", "engine identity"),
        )
        for index, (contents, message) in enumerate(invalid_headers):
            with self.subTest(contents=contents):
                path = os.path.join(self.root, "invalid-header-%d" % index)
                Path(path).write_text(contents, encoding="utf-8")
                with self.assertRaisesRegex(ramdisk.RamdiskError, message):
                    ramdisk._usage_read(path)

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

    def test_atomic_json_stream_close_cannot_mask_primary_error(self):
        stream = mock.MagicMock()
        stream.__enter__.return_value = stream
        stream.__exit__.side_effect = OSError("secondary stream close")
        stream.close.side_effect = OSError("secondary stream close")
        with mock.patch.object(
            state_support.os,
            "fdopen",
            return_value=stream,
        ), mock.patch.object(
            state_support.json,
            "dump",
            side_effect=ValueError("primary JSON serialization"),
        ):
            with self.assertRaisesRegex(
                ValueError,
                "primary JSON serialization",
            ):
                state_support._atomic_json(
                    os.path.join(self.root, "atomic.json"),
                    {"unsafe": object()},
                )

    @requires_native_dirfd
    def test_atomic_temp_creation_stays_inside_bound_parent(self):
        parent = os.path.join(self.root, "usage-parent")
        original = os.path.join(self.root, "usage-parent-original")
        replacement = os.path.join(self.root, "usage-parent-replacement")
        os.makedirs(parent)
        os.makedirs(replacement)
        target = os.path.join(parent, ".coli_usage")
        real_open = state_support.os.open
        swapped = False

        def swap_parent_at_temp_open(path, flags, *args, **kwargs):
            nonlocal swapped
            if (
                isinstance(path, str)
                and path.startswith(".usage-")
                and kwargs.get("dir_fd") is not None
                and not swapped
            ):
                swapped = True
                os.rename(parent, original)
                os.rename(replacement, parent)
            return real_open(path, flags, *args, **kwargs)

        with mock.patch.object(
            state_support.os,
            "open",
            side_effect=swap_parent_at_temp_open,
        ):
            with self.assertRaisesRegex(
                ramdisk.RamdiskError,
                "parent identity changed",
            ):
                state_support._usage_write(
                    target,
                    {"0:1": 1},
                    require_native=True,
                )

        self.assertTrue(swapped)
        self.assertEqual(list(Path(original).glob(".usage-*")), [])
        self.assertEqual(list(Path(parent).glob(".usage-*")), [])

    def test_usage_write_stream_close_cannot_mask_primary_error(self):
        stream = mock.MagicMock()
        stream.__enter__.return_value = stream
        stream.__exit__.side_effect = OSError("secondary stream close")
        stream.close.side_effect = OSError("secondary stream close")
        path = os.path.join(self.root, ".coli_usage")
        with mock.patch.object(
            state_support.os,
            "fdopen",
            return_value=stream,
        ):
            stream.write.side_effect = ValueError("primary usage write")
            with self.assertRaisesRegex(ValueError, "primary usage write"):
                state_support._usage_write(path, {"0:1": 1})

    def test_managed_usage_close_cannot_mask_parse_error(self):
        state_usage = os.path.join(self.root, ".coli_usage")
        Path(state_usage).write_text("-1 malformed\n", encoding="utf-8")
        real_close = state_support.os.close

        def fail_after_close(descriptor):
            real_close(descriptor)
            raise OSError("secondary descriptor close")

        with mock.patch.object(
            state_support.os,
            "close",
            side_effect=fail_after_close,
        ):
            with self.assertRaisesRegex(
                ramdisk.RamdiskError,
                "malformed usage header",
            ):
                state_support._managed_usage_read(state_usage)

    def test_directory_fsync_error_propagates_and_closes_descriptor(self):
        with mock.patch.object(
            state_support.os,
            "open",
            return_value=731,
        ), mock.patch.object(
            state_support.os,
            "fsync",
            side_effect=OSError(5, "synthetic directory EIO"),
        ), mock.patch.object(state_support.os, "close") as close:
            with self.assertRaisesRegex(OSError, "synthetic directory EIO"):
                _REAL_FSYNC_DIRECTORY(self.root)

        close.assert_called_once_with(731)

    def test_directory_fsync_preserves_primary_error_over_close_error(self):
        with mock.patch.object(
            state_support.os,
            "open",
            return_value=732,
        ), mock.patch.object(
            state_support.os,
            "fsync",
            side_effect=OSError(5, "primary directory EIO"),
        ), mock.patch.object(
            state_support.os,
            "close",
            side_effect=OSError(9, "secondary close failure"),
        ) as close:
            with self.assertRaisesRegex(OSError, "primary directory EIO"):
                _REAL_FSYNC_DIRECTORY(self.root)

        close.assert_called_once_with(732)

    def test_directory_fsync_reports_close_error_after_successful_sync(self):
        with mock.patch.object(
            state_support.os,
            "open",
            return_value=733,
        ), mock.patch.object(
            state_support.os,
            "fsync",
        ), mock.patch.object(
            state_support.os,
            "close",
            side_effect=OSError(9, "directory close failure"),
        ) as close:
            with self.assertRaisesRegex(OSError, "directory close failure"):
                _REAL_FSYNC_DIRECTORY(self.root)

        close.assert_called_once_with(733)

    def test_pending_manifest_save_reports_unproven_directory_commit(self):
        path = os.path.join(self.root, "pending-manifest.json")
        pending = {
            "state": "starting",
            "pending_launches": [{"operation_id": "start:" + ("a" * 32)}],
        }
        durability_error = OSError(5, "synthetic directory EIO")
        with mock.patch.object(
            state_support,
            "_fsync_bound_directory",
            side_effect=durability_error,
        ), mock.patch.object(
            state_support,
            "_fsync_directory",
            side_effect=durability_error,
        ):
            with self.assertRaisesRegex(OSError, "synthetic directory EIO"):
                state_support._save_manifest(
                    pending,
                    manifest_path=lambda: path,
                )

        persisted = json.loads(Path(path).read_text(encoding="utf-8"))
        self.assertEqual(
            persisted["pending_launches"][0]["operation_id"],
            "start:" + ("a" * 32),
        )

    def test_uncertain_canonical_marker_keeps_journal_for_retry(self):
        model_dir = os.path.join(self.root, "model")
        canonical = os.path.join(model_dir, ".coli_usage")
        state_dir = os.path.join(self.root, "node-state")
        os.makedirs(model_dir)
        os.makedirs(state_dir)
        ramdisk._usage_write(canonical, {"0:1": 10})
        merge_id = "f" * 32
        delta_path = os.path.join(state_dir, ".coli_usage.delta.json")
        ramdisk._atomic_json(
            delta_path,
            {"version": 1, "id": merge_id, "delta": {"0:1": 2}},
        )

        durability_error = OSError(5, "synthetic directory EIO")
        with mock.patch.object(
            state_support,
            "_fsync_bound_directory",
            side_effect=durability_error,
        ), mock.patch.object(
            state_support,
            "_fsync_directory",
            side_effect=durability_error,
        ):
            with self.assertRaisesRegex(OSError, "synthetic directory EIO"):
                ramdisk._recover_delta(state_dir, canonical)

        self.assertTrue(os.path.exists(delta_path))
        self.assertEqual(ramdisk._usage_read(canonical)["0:1"], 12)
        self.assertIn(merge_id, ramdisk._usage_merge_ids(canonical))

        ramdisk._recover_delta(state_dir, canonical)
        self.assertFalse(os.path.exists(delta_path))
        self.assertEqual(ramdisk._usage_read(canonical)["0:1"], 12)

    @requires_native_dirfd
    def test_existing_marker_reproves_canonical_parent_before_journal_unlink(self):
        model_dir = os.path.join(self.root, "model")
        state_dir = os.path.join(self.root, "node-state")
        os.makedirs(model_dir)
        os.makedirs(state_dir)
        canonical = os.path.join(model_dir, ".coli_usage")
        merge_id = "0" * 32
        delta_path = os.path.join(state_dir, ".coli_usage.delta.json")
        ramdisk._usage_write(canonical, {"0:1": 12}, merge_id=merge_id)
        ramdisk._atomic_json(
            delta_path,
            {"version": 1, "id": merge_id, "delta": {"0:1": 2}},
        )
        canonical_before = Path(canonical).read_bytes()
        synced = []
        real_sync = state_support._fsync_bound_directory
        canonical_parent = state_support._stat_identity(os.stat(model_dir))
        managed_parent = state_support._stat_identity(os.stat(state_dir))

        def fail_canonical_sync(descriptor):
            info = os.fstat(descriptor)
            identity = (info.st_dev, info.st_ino)
            if identity == canonical_parent:
                raise OSError("canonical parent durability uncertain")
            return real_sync(descriptor)

        with mock.patch.object(
            state_support,
            "_fsync_bound_directory",
            side_effect=fail_canonical_sync,
        ):
            with self.assertRaisesRegex(
                OSError,
                "canonical parent durability uncertain",
            ):
                ramdisk._recover_delta(state_dir, canonical)

        self.assertTrue(os.path.exists(delta_path))
        self.assertEqual(Path(canonical).read_bytes(), canonical_before)

        def track_sync(descriptor):
            info = os.fstat(descriptor)
            synced.append((info.st_dev, info.st_ino))
            return real_sync(descriptor)

        with mock.patch.object(
            state_support,
            "_fsync_bound_directory",
            side_effect=track_sync,
        ):
            ramdisk._recover_delta(state_dir, canonical)

        self.assertIn(canonical_parent, synced)
        self.assertIn(managed_parent, synced)
        self.assertLess(
            synced.index(canonical_parent),
            synced.index(managed_parent),
        )
        self.assertFalse(os.path.exists(delta_path))

    def test_durable_unlink_reports_unproven_directory_commit(self):
        path = os.path.join(self.root, "journal.json")
        Path(path).write_text("{}\n", encoding="utf-8")
        with mock.patch.object(
            state_support,
            "_fsync_directory",
            side_effect=(
                OSError(5, "synthetic directory EIO"),
                None,
            ),
        ) as sync_directory:
            with self.assertRaisesRegex(OSError, "synthetic directory EIO"):
                state_support._durable_unlink(path)
            state_support._durable_unlink(path)

        self.assertFalse(os.path.exists(path))
        self.assertEqual(sync_directory.call_count, 2)
        sync_directory.assert_has_calls(
            [mock.call(self.root), mock.call(self.root)]
        )

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

    def test_pending_operation_id_must_bind_usage_transaction(self):
        pending = {
            "operation_id": "start:" + ("1" * 32),
            "nonce": "2" * 48,
            "uid": host_uid(),
            "port": 8000,
            "node": None,
            "state_dir": self.recovery_state_dir(),
            "weights_dir": "/mnt/colibri-test",
            "launch_not_before": 100,
            "launcher_pid": 700,
            "launcher_starttime": 90,
            "launcher_cmdline": ["coli", "ramdisk", "start"],
            "expected_command": ["coli", "serve"],
            "usage_baseline": {"0:1": 3},
            "usage_merge_id": "3" * 32,
        }
        manifest = self.manifest(state="starting")
        manifest["pending_launches"] = [pending]
        ramdisk._atomic_json(ramdisk._manifest_path(), manifest)

        with self.assertRaisesRegex(
            ramdisk.RamdiskError,
            "pending launch recovery",
        ):
            ramdisk._load_manifest(required=True)

    def test_usage_transaction_ids_are_unique_across_all_authorities(self):
        mounts = [
            "/mnt/colibri-test/node0",
            "/mnt/colibri-test/node1",
            "/mnt/colibri-test/node2",
        ]
        base = self.manifest(
            state="error",
            mount_paths=mounts,
            processes=[{"pid": 12020}],
        )
        base["processes"][0]["usage_merge_id"] = "4" * 32
        base["pending_launches"] = [
            {
                "operation_id": "start:" + ("5" * 32),
                "nonce": "6" * 48,
                "uid": host_uid(),
                "port": 8001,
                "node": 1,
                "state_dir": self.recovery_state_dir(node=1),
                "weights_dir": mounts[1],
                "launch_not_before": 100,
                "launcher_pid": 700,
                "launcher_starttime": 90,
                "launcher_cmdline": ["coli", "ramdisk", "start"],
                "expected_command": ["coli", "serve"],
                "usage_baseline": {"0:1": 3},
                "usage_merge_id": "5" * 32,
            }
        ]
        base["recovery"] = {
            "operation": "start",
            "state": "attention-required",
            "retained_processes": [
                {
                    "pid": 12022,
                    "pgid": 12022,
                    "node": 2,
                    "state_dir": self.recovery_state_dir(node=2),
                    "usage_baseline": {"0:1": 3},
                    "usage_merge_id": "7" * 32,
                    "error": "group absence unproven",
                }
            ],
        }

        ramdisk._atomic_json(ramdisk._manifest_path(), base)
        loaded = ramdisk._load_manifest(required=True)
        self.assertEqual(loaded["processes"][0]["usage_merge_id"], "4" * 32)

        duplicate_cases = (
            ("process-pending", ("pending_launches", 0), "4" * 32),
            (
                "process-retained",
                ("recovery", "retained_processes", 0),
                "4" * 32,
            ),
            (
                "pending-retained",
                ("recovery", "retained_processes", 0),
                "5" * 32,
            ),
        )
        for label, path, duplicate in duplicate_cases:
            with self.subTest(label=label):
                manifest = copy.deepcopy(base)
                target = manifest
                for key in path:
                    target = target[key]
                target["usage_merge_id"] = duplicate
                if path[0] == "pending_launches":
                    target["operation_id"] = "start:" + duplicate
                ramdisk._atomic_json(ramdisk._manifest_path(), manifest)
                with self.assertRaisesRegex(
                    ramdisk.RamdiskError,
                    "duplicate usage transaction",
                ):
                    ramdisk._load_manifest(required=True)

        legacy = copy.deepcopy(base)
        legacy["processes"][0].pop("usage_merge_id")
        ramdisk._atomic_json(ramdisk._manifest_path(), legacy)
        self.assertIsNone(
            ramdisk._load_manifest(required=True)["processes"][0].get(
                "usage_merge_id"
            )
        )

    def test_manifest_rejects_untrusted_process_usage_merge_marker(self):
        cases = (
            (True, "e" * 32),
            ("not-a-timestamp", "e" * 32),
            ("2026-08-01T00:00:00Z", None),
        )
        for marker, merge_id in cases:
            with self.subTest(marker=marker, merge_id=merge_id):
                manifest = self.manifest(
                    state="stopped",
                    processes=[
                        {
                            "pid": 12347,
                            "usage_merge_id": merge_id,
                            "usage_merged_at": marker,
                        }
                    ],
                )
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

    def test_outcome_unknown_pending_launch_is_reconciled_only_by_stop(self):
        pending = {
            "operation_id": "start:" + ("6" * 32),
            "nonce": "7" * 48,
            "uid": host_uid(),
            "port": 8000,
            "node": None,
            "state_dir": self.recovery_state_dir(),
            "weights_dir": "/mnt/colibri-test",
            "launch_not_before": 100,
            "launcher_pid": 700,
            "launcher_starttime": 90,
            "launcher_cmdline": ["coli", "ramdisk", "start"],
            "expected_command": ["coli", "serve"],
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
            ramdisk,
            "_managed_launch_processes",
            side_effect=[[], []],
        ) as discover, mock.patch.object(
            ramdisk, "_terminate_verified_group"
        ) as terminate, mock.patch.object(
            ramdisk, "_merge_usage"
        ) as merge:
            stopped = ramdisk.stop()
        terminate.assert_not_called()
        self.assertEqual(discover.call_count, 2)
        merge.assert_called_once()
        self.assertEqual(stopped["state"], "stopped")
        self.assertEqual(stopped["pending_launches"], [])
        persisted = ramdisk._load_manifest(required=True)
        self.assertEqual(persisted["state"], "stopped")
        self.assertEqual(persisted["pending_launches"], [])

    def test_pending_live_group_identity_is_persisted_before_stop_signals(self):
        pending = {
            "operation_id": "start:" + ("8" * 32),
            "nonce": "9" * 48,
            "uid": host_uid(),
            "port": 8000,
            "node": None,
            "state_dir": self.recovery_state_dir(),
            "weights_dir": "/mnt/colibri-test",
            "launch_not_before": 100,
            "launcher_pid": 700,
            "launcher_starttime": 90,
            "launcher_cmdline": ["coli", "ramdisk", "start"],
            "expected_command": ["coli", "serve"],
            "usage_baseline": {"0:1": 3},
            "usage_merge_id": "8" * 32,
        }
        manifest = self.manifest(state="starting")
        manifest["pending_launches"] = [copy.deepcopy(pending)]
        ramdisk._atomic_json(ramdisk._manifest_path(), manifest)
        candidate = {
            "pid": 12010,
            "uid": host_uid(),
            "starttime": 400,
            "nonce": pending["nonce"],
            "pgid": 12010,
            "sid": 12010,
            "state_dir": pending["state_dir"],
            "weights_dir": pending["weights_dir"],
            "cmdline": ["coli", "serve"],
        }

        def terminate(record):
            durable = ramdisk._load_manifest(required=True)
            observed = durable["pending_launches"][0]["observed_group"]
            self.assertEqual(observed["pgid"], 12010)
            self.assertEqual(observed["leader_starttime"], 400)
            self.assertEqual(record["starttime"], 400)
            return None

        with mock.patch.object(
            ramdisk,
            "_managed_launch_processes",
            side_effect=[[candidate], []],
        ), mock.patch.object(
            ramdisk,
            "_process_group_members",
            return_value=([candidate], []),
        ), mock.patch.object(
            ramdisk,
            "_process_matches",
            return_value=(True, "running", candidate),
        ), mock.patch.object(
            ramdisk, "_group_alive", return_value=False
        ), mock.patch.object(
            ramdisk, "_terminate_verified_group", side_effect=terminate
        ) as terminate_group, mock.patch.object(
            ramdisk, "_merge_usage"
        ) as merge:
            stopped = ramdisk.stop()

        terminate_group.assert_called_once()
        merge.assert_called_once()
        self.assertEqual(stopped["state"], "stopped")
        self.assertEqual(stopped["pending_launches"], [])

    def test_manifest_rejects_untrusted_pending_usage_merge_timestamp(self):
        pending = {
            "operation_id": "start:" + ("4" * 32),
            "nonce": "5" * 48,
            "uid": host_uid(),
            "port": 8000,
            "node": None,
            "state_dir": self.recovery_state_dir(),
            "weights_dir": "/mnt/colibri-test",
            "launch_not_before": 100,
            "launcher_pid": 700,
            "launcher_starttime": 90,
            "launcher_cmdline": ["coli", "ramdisk", "start"],
            "expected_command": ["coli", "serve"],
            "usage_baseline": {"0:1": 3},
            "usage_merge_id": "4" * 32,
        }
        for marker in (True, "not-a-timestamp"):
            with self.subTest(marker=marker):
                manifest = self.manifest(state="starting")
                manifest["pending_launches"] = [
                    dict(pending, usage_merged_at=marker)
                ]
                ramdisk._atomic_json(ramdisk._manifest_path(), manifest)
                with self.assertRaisesRegex(
                    ramdisk.RamdiskError,
                    "unsafe pending launch recovery",
                ):
                    ramdisk._load_manifest(required=True)

    def test_pending_wrapper_dead_group_uses_member_verified_pgid(self):
        pending = {
            "operation_id": "start:" + ("a" * 32),
            "nonce": "b" * 48,
            "uid": host_uid(),
            "port": 8000,
            "node": None,
            "state_dir": self.recovery_state_dir(),
            "weights_dir": "/mnt/colibri-test",
            "launch_not_before": 100,
            "launcher_pid": 700,
            "launcher_starttime": 90,
            "launcher_cmdline": ["coli", "ramdisk", "start"],
            "expected_command": ["coli", "serve"],
            "usage_baseline": {"0:1": 3},
            "usage_merge_id": "a" * 32,
            "observed_group": {
                "pgid": 12011,
                "uid": host_uid(),
                "leader_starttime": 401,
            },
        }
        manifest = self.manifest(state="starting")
        manifest["pending_launches"] = [copy.deepcopy(pending)]
        ramdisk._atomic_json(ramdisk._manifest_path(), manifest)
        child = {
            "pid": 12012,
            "uid": host_uid(),
            "starttime": 402,
            "nonce": pending["nonce"],
            "pgid": 12011,
            "sid": 12011,
            "state_dir": pending["state_dir"],
            "weights_dir": pending["weights_dir"],
            "cmdline": ["colibri"],
        }

        def terminate(record):
            self.assertEqual(record["pid"], 12011)
            self.assertEqual(record["pgid"], 12011)
            self.assertIsNone(record["starttime"])
            durable = ramdisk._load_manifest(required=True)
            self.assertEqual(
                durable["pending_launches"][0]["observed_group"]
                ["leader_starttime"],
                401,
            )
            return None

        with mock.patch.object(
            ramdisk,
            "_managed_launch_processes",
            side_effect=[[child], []],
        ), mock.patch.object(
            ramdisk,
            "_process_group_members",
            return_value=([child], []),
        ), mock.patch.object(
            ramdisk,
            "_process_matches",
            return_value=(True, "running-group", {"members": [child]}),
        ), mock.patch.object(
            ramdisk, "_group_alive", return_value=False
        ), mock.patch.object(
            ramdisk, "_terminate_verified_group", side_effect=terminate
        ) as terminate_group, mock.patch.object(
            ramdisk, "_merge_usage"
        ) as merge:
            stopped = ramdisk.stop()

        terminate_group.assert_called_once()
        merge.assert_called_once()
        self.assertEqual(stopped["state"], "stopped")

    def test_pending_preflight_accepts_the_exact_inert_zombie_leader(self):
        pending = {
            "operation_id": "start:" + ("a" * 32),
            "nonce": "b" * 48,
            "uid": host_uid(),
            "port": 8000,
            "node": None,
            "state_dir": self.recovery_state_dir(),
            "weights_dir": "/mnt/colibri-test",
            "launch_not_before": 100,
            "launcher_pid": 700,
            "launcher_starttime": 90,
            "launcher_cmdline": ["coli", "ramdisk", "start"],
            "expected_command": ["coli", "serve"],
            "usage_baseline": {"0:1": 3},
            "usage_merge_id": "a" * 32,
        }
        zombie = {
            "pid": 12011,
            "uid": host_uid(),
            "state": "Z",
            "inert": True,
            "starttime": 401,
            "nonce": None,
            "pgid": 12011,
            "sid": 12011,
            "state_dir": None,
            "weights_dir": None,
        }
        child = {
            "pid": 12012,
            "uid": host_uid(),
            "state": "S",
            "inert": False,
            "starttime": 402,
            "nonce": pending["nonce"],
            "pgid": 12011,
            "sid": 12011,
            "state_dir": pending["state_dir"],
            "weights_dir": pending["weights_dir"],
            "cmdline": ["colibri"],
        }

        def process_matches(record):
            self.assertEqual(record["pid"], 12011)
            self.assertEqual(record["starttime"], 401)
            return True, "running-group", {"members": [zombie, child]}

        preflights, failures = lifecycle_support._preflight_pending_launches(
            {"pending_launches": [pending]},
            discover_managed_launches=mock.Mock(return_value=[child]),
            process_matches=process_matches,
            process_group_members=mock.Mock(return_value=([zombie, child], [])),
            group_alive=mock.Mock(return_value=True),
        )

        self.assertEqual(failures, [])
        self.assertEqual(preflights[0]["observed_group"]["leader_starttime"], 401)

    def test_pending_preflight_accepts_exact_inert_nonleader(self):
        pending = {
            "operation_id": "start:" + ("a" * 32),
            "nonce": "b" * 48,
            "uid": host_uid(),
            "port": 8000,
            "node": None,
            "state_dir": self.recovery_state_dir(),
            "weights_dir": "/mnt/colibri-test",
            "launch_not_before": 100,
            "launcher_pid": 700,
            "launcher_starttime": 90,
            "launcher_cmdline": ["coli", "ramdisk", "start"],
            "expected_command": ["coli", "serve"],
            "usage_baseline": {"0:1": 3},
            "usage_merge_id": "a" * 32,
        }
        leader = {
            "pid": 12011,
            "uid": host_uid(),
            "inert": False,
            "starttime": 401,
            "nonce": pending["nonce"],
            "pgid": 12011,
            "sid": 12011,
            "state_dir": pending["state_dir"],
            "weights_dir": pending["weights_dir"],
            "cmdline": ["coli", "serve"],
        }
        zombie = {
            "pid": 12012,
            "uid": host_uid(),
            "state": "Z",
            "inert": True,
            "starttime": 402,
            "nonce": None,
            "pgid": 12011,
            "sid": 12011,
            "state_dir": None,
            "weights_dir": None,
        }
        process_matches = mock.Mock(
            return_value=(True, "running", {"members": [leader, zombie]})
        )

        preflights, failures = lifecycle_support._preflight_pending_launches(
            {"pending_launches": [pending]},
            discover_managed_launches=mock.Mock(return_value=[leader]),
            process_matches=process_matches,
            process_group_members=mock.Mock(
                return_value=([leader, zombie], [])
            ),
            group_alive=mock.Mock(return_value=True),
        )

        self.assertEqual(failures, [])
        self.assertEqual(preflights[0]["record"]["starttime"], 401)
        process_matches.assert_called_once()

    def test_pending_preflight_rejects_mismatched_inert_nonleader(self):
        pending = {
            "operation_id": "start:" + ("a" * 32),
            "nonce": "b" * 48,
            "uid": host_uid(),
            "port": 8000,
            "node": None,
            "state_dir": self.recovery_state_dir(),
            "weights_dir": "/mnt/colibri-test",
            "launch_not_before": 100,
            "launcher_pid": 700,
            "launcher_starttime": 90,
            "launcher_cmdline": ["coli", "ramdisk", "start"],
            "expected_command": ["coli", "serve"],
            "usage_baseline": {"0:1": 3},
            "usage_merge_id": "a" * 32,
        }
        leader = {
            "pid": 12011,
            "uid": host_uid(),
            "inert": False,
            "starttime": 401,
            "nonce": pending["nonce"],
            "pgid": 12011,
            "sid": 12011,
            "state_dir": pending["state_dir"],
            "weights_dir": pending["weights_dir"],
            "cmdline": ["coli", "serve"],
        }
        zombie = {
            "pid": 12012,
            "uid": host_uid(),
            "state": "Z",
            "inert": True,
            "starttime": 402,
            "nonce": None,
            "pgid": 12011,
            "sid": 12011,
            "state_dir": None,
            "weights_dir": None,
        }
        for field, value in (
            ("uid", host_uid() + 1),
            ("pgid", 12099),
            ("sid", 12099),
        ):
            with self.subTest(field=field):
                process_matches = mock.Mock()
                mismatched = dict(zombie, **{field: value})
                preflights, failures = (
                    lifecycle_support._preflight_pending_launches(
                        {"pending_launches": [pending]},
                        discover_managed_launches=mock.Mock(
                            return_value=[leader]
                        ),
                        process_matches=process_matches,
                        process_group_members=mock.Mock(
                            return_value=([leader, mismatched], [])
                        ),
                        group_alive=mock.Mock(return_value=True),
                    )
                )

                self.assertEqual(preflights, [])
                self.assertRegex(failures[0], "foreign or mismatched member")
                process_matches.assert_not_called()

    def test_pending_launch_ambiguous_groups_refuse_without_side_effects(self):
        pending = {
            "operation_id": "start:" + ("c" * 32),
            "nonce": "d" * 48,
            "uid": host_uid(),
            "port": 8000,
            "node": None,
            "state_dir": self.recovery_state_dir(),
            "weights_dir": "/mnt/colibri-test",
            "launch_not_before": 100,
            "launcher_pid": 700,
            "launcher_starttime": 90,
            "launcher_cmdline": ["coli", "ramdisk", "start"],
            "expected_command": ["coli", "serve"],
            "usage_baseline": {"0:1": 3},
            "usage_merge_id": "c" * 32,
        }
        manifest = self.manifest(state="starting")
        manifest["pending_launches"] = [copy.deepcopy(pending)]
        ramdisk._atomic_json(ramdisk._manifest_path(), manifest)

        def candidate(pid, pgid):
            return {
                "pid": pid,
                "uid": host_uid(),
                "starttime": 500 + pid,
                "nonce": pending["nonce"],
                "pgid": pgid,
                "sid": pgid,
                "state_dir": pending["state_dir"],
                "weights_dir": pending["weights_dir"],
                "cmdline": ["colibri"],
            }

        with mock.patch.object(
            ramdisk,
            "_managed_launch_processes",
            return_value=[candidate(12020, 12020), candidate(12021, 12021)],
        ), mock.patch.object(
            ramdisk, "_terminate_verified_group"
        ) as terminate, mock.patch.object(
            ramdisk, "_merge_usage"
        ) as merge:
            with self.assertRaisesRegex(
                ramdisk.RamdiskError,
                "multiple process groups",
            ):
                ramdisk.stop()

        terminate.assert_not_called()
        merge.assert_not_called()
        persisted = ramdisk._load_manifest(required=True)
        self.assertEqual(len(persisted["pending_launches"]), 1)
        with mock.patch.object(ramdisk, "_mount_at", return_value=None):
            recovery_status = ramdisk.status(deep=False)
        self.assertNotIn(
            pending["usage_merge_id"],
            json.dumps(recovery_status, sort_keys=True),
        )

        with mock.patch.object(
            ramdisk,
            "_managed_launch_processes",
            side_effect=[[], []],
        ), mock.patch.object(ramdisk, "_merge_usage"):
            stopped = ramdisk.stop()
        self.assertEqual(stopped["state"], "stopped")
        self.assertNotIn("cleanup_errors", stopped)

    def test_pending_launch_waits_for_ordinary_process_global_preflight(self):
        pending = {
            "operation_id": "start:" + ("e" * 32),
            "nonce": "f" * 48,
            "uid": host_uid(),
            "port": 8001,
            "node": 1,
            "state_dir": os.path.join(self.root, "pending-state"),
            "weights_dir": "/mnt/colibri-test/node1",
            "launch_not_before": 100,
            "launcher_pid": 700,
            "launcher_starttime": 90,
            "launcher_cmdline": ["coli", "ramdisk", "start"],
            "expected_command": ["coli", "serve"],
            "usage_baseline": {"0:1": 3},
            "usage_merge_id": "e" * 32,
        }
        candidate = {
            "pid": 12030,
            "uid": host_uid(),
            "starttime": 530,
            "nonce": pending["nonce"],
            "pgid": 12030,
            "sid": 12030,
            "state_dir": pending["state_dir"],
            "weights_dir": pending["weights_dir"],
            "cmdline": ["colibri"],
        }
        ordinary = {
            "pid": 12031,
            "pgid": 12031,
            "uid": host_uid(),
            "nonce": "1" * 48,
            "state_dir": os.path.join(self.root, "ordinary-state"),
            "usage_baseline": {},
        }
        manifest = {
            "state": "starting",
            "plan": {
                "model": {"path": os.path.join(self.root, "model")},
                "mounts": [
                    {"node": 0, "path": "/mnt/colibri-test/node0"},
                    {"node": 1, "path": pending["weights_dir"]},
                ],
            },
            "mounts": [
                {"node": 0, "path": "/mnt/colibri-test/node0"},
                {"node": 1, "path": pending["weights_dir"]},
            ],
            "processes": [ordinary],
            "pending_launches": [pending],
        }
        process_matches = mock.Mock(
            side_effect=[
                (True, "running", candidate),
                (False, "foreign-uid", None),
            ]
        )
        terminate = mock.Mock()
        merge = mock.Mock()

        with self.assertRaisesRegex(
            ramdisk.RamdiskError,
            "unverified processes",
        ):
            lifecycle_support.stop(
                load_manifest=mock.Mock(return_value=manifest),
                discover_managed_launches=mock.Mock(return_value=[candidate]),
                process_matches=process_matches,
                process_group_members=mock.Mock(return_value=([candidate], [])),
                group_alive=mock.Mock(),
                managed_child_liveness=mock.Mock(return_value=None),
                save_manifest=mock.Mock(),
                terminate_verified_group=terminate,
                merge_usage=merge,
                bind_usage_transaction=mock.Mock(),
            )

        terminate.assert_not_called()
        merge.assert_not_called()
        self.assertNotIn("observed_group", pending)

    def test_pending_launch_retry_does_not_repeat_merged_transaction(self):
        pending = {
            "operation_id": "start:" + ("2" * 32),
            "nonce": "3" * 48,
            "uid": host_uid(),
            "port": 8000,
            "node": None,
            "state_dir": self.recovery_state_dir(),
            "weights_dir": "/mnt/colibri-test",
            "launch_not_before": 100,
            "launcher_pid": 700,
            "launcher_starttime": 90,
            "launcher_cmdline": ["coli", "ramdisk", "start"],
            "expected_command": ["coli", "serve"],
            "usage_baseline": {"0:1": 3},
            "usage_merge_id": "2" * 32,
        }
        manifest = self.manifest(state="starting")
        manifest["pending_launches"] = [copy.deepcopy(pending)]
        ramdisk._atomic_json(ramdisk._manifest_path(), manifest)
        real_save = ramdisk._save_manifest
        save_calls = []

        def fail_pending_removal(value):
            save_calls.append(copy.deepcopy(value))
            if len(save_calls) == 2:
                raise OSError("pending removal save failed")
            return real_save(value)

        applied_transactions = set()
        applications = []

        def idempotent_merge(record, _canonical, plan=None):
            del plan
            merge_id = record["usage_merge_id"]
            if merge_id not in applied_transactions:
                applied_transactions.add(merge_id)
                applications.append(merge_id)

        merge = mock.Mock(side_effect=idempotent_merge)
        with mock.patch.object(
            ramdisk,
            "_managed_launch_processes",
            side_effect=[[], []],
        ), mock.patch.object(
            ramdisk, "_merge_usage", merge
        ), mock.patch.object(
            ramdisk, "_save_manifest", side_effect=fail_pending_removal
        ):
            with self.assertRaisesRegex(
                ramdisk.RamdiskError,
                "pending authority could not be removed",
            ):
                ramdisk.stop()

        after_failure = ramdisk._load_manifest(required=True)
        self.assertEqual(len(after_failure["pending_launches"]), 1)
        self.assertIn("usage_merged_at", after_failure["pending_launches"][0])
        merge.assert_called_once()
        self.assertEqual(applications, [pending["usage_merge_id"]])

        with mock.patch.object(
            ramdisk,
            "_managed_launch_processes",
            side_effect=[[], []],
        ), mock.patch.object(ramdisk, "_merge_usage", merge):
            stopped = ramdisk.stop()

        self.assertEqual(merge.call_count, 2)
        self.assertEqual(applications, [pending["usage_merge_id"]])
        self.assertEqual(stopped["state"], "stopped")
        self.assertEqual(stopped["pending_launches"], [])

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
            "authority_version": lifecycle_support._RETAINED_AUTHORITY_VERSION,
            "nonce": "a" * 48,
            "uid": 1000,
            "weights_dir": os.path.join(self.root, "accounting-weights"),
            "launch_not_before": 1,
            "launcher_pid": 1,
            "launcher_starttime": 1,
            "launcher_cmdline": ["coli"],
            "expected_command": ["coli", "serve"],
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
                discover_managed_launches=mock.Mock(return_value=[]),
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
            discover_managed_launches=mock.Mock(return_value=[]),
            merge_usage=merge,
            save_manifest=lambda ignored: None,
        )

        self.assertEqual(state_support._usage_read(canonical), {"0:1": 12})
        self.assertEqual(
            restarted["recovery"]["retained_processes"],
            [],
        )

    def _retained_entry_with_authority(
        self,
        state_dir,
        merge_id,
        *,
        pid=12009,
        nonce="a" * 48,
    ):
        """A retained unpublished entry carrying full discovery authority."""
        return {
            "pid": pid,
            "pgid": pid,
            "node": None,
            "state_dir": state_dir,
            "usage_baseline": {"0:1": 10},
            "usage_merge_id": merge_id,
            "error": "group absence unproven",
            "authority_version": lifecycle_support._RETAINED_AUTHORITY_VERSION,
            "nonce": nonce,
            "uid": 1000,
            "weights_dir": os.path.join(self.root, "authority-weights"),
            "launch_not_before": 1,
            "launcher_pid": 1,
            "launcher_starttime": 1,
            "launcher_cmdline": ["coli"],
            "expected_command": ["coli", "serve"],
        }

    def _unpublished_usage_manifest(self, merge_id, baseline_delta=2):
        model_dir = os.path.join(self.root, "unpub-model-%s" % merge_id)
        state_dir = os.path.join(self.root, "unpub-state-%s" % merge_id)
        os.makedirs(model_dir)
        os.makedirs(state_dir)
        canonical = os.path.join(model_dir, ".coli_usage")
        state_support._usage_write(canonical, {"0:1": 10})
        state_support._usage_write(
            os.path.join(state_dir, ".coli_usage"),
            {"0:1": 10 + baseline_delta},
        )
        entry = self._retained_entry_with_authority(state_dir, merge_id)
        manifest = {
            "state": "error",
            "plan": {"model": {"path": model_dir}, "mounts": []},
            "recovery": {
                "operation": "start",
                "state": "attention-required",
                "retained_processes": [entry],
            },
        }
        return manifest, canonical, entry

    def _unpublished_merge(self, canonical):
        def merge(record, canonical_path, plan=None):
            return state_support._merge_usage(
                record,
                canonical_path,
                plan=plan,
                filesystem_for_path=lambda ignored: "ext4",
                source_still_matches=lambda ignored: None,
            )
        return merge

    def test_retained_authority_helper_preserves_discovery_metadata(self):
        pending_entry = {
            "nonce": "deadbeef" * 6,
            "uid": 1000,
            "weights_dir": "/srv/w",
            "launch_not_before": 12345,
            "launcher_pid": 99,
            "launcher_starttime": 4321,
            "launcher_cmdline": ["coli", "ramdisk"],
            "expected_command": ["coli", "serve"],
        }
        authority = lifecycle_support._retained_process_authority(pending_entry)
        self.assertEqual(
            authority["authority_version"],
            lifecycle_support._RETAINED_AUTHORITY_VERSION,
        )
        entry = dict(
            pid=12010,
            pgid=12010,
            state_dir="/srv/state",
            usage_baseline={"0:1": 1},
            usage_merge_id="c" * 32,
            **authority,
        )
        validated = lifecycle_support._retained_authority(entry)
        self.assertEqual(validated["nonce"], pending_entry["nonce"])
        self.assertEqual(validated["uid"], pending_entry["uid"])
        self.assertEqual(
            validated["weights_dir"], pending_entry["weights_dir"]
        )
        self.assertEqual(
            validated["launch_not_before"],
            pending_entry["launch_not_before"],
        )
        self.assertEqual(
            validated["launcher_pid"], pending_entry["launcher_pid"]
        )
        # Legacy records (no persisted authority) cannot be validated, so
        # callers fail closed instead of guessing absence.
        legacy = {
            "pid": 1,
            "pgid": 1,
            "state_dir": "/s",
            "usage_baseline": {},
            "usage_merge_id": "d" * 32,
        }
        self.assertIsNone(lifecycle_support._retained_authority(legacy))

    def test_unpublished_recovery_refuses_without_discovery_authority(self):
        # A legacy retained record (created before authority was persisted)
        # cannot be positively proven absent even when the original group is
        # gone: refuse to merge, keep it retained, mutate no accounting.
        merge_id = "6" * 32
        manifest, canonical, _entry = self._unpublished_usage_manifest(merge_id)
        authority_keys = (
            "authority_version",
            "nonce",
            "uid",
            "weights_dir",
            "launch_not_before",
            "launcher_pid",
            "launcher_starttime",
            "launcher_cmdline",
            "expected_command",
        )
        manifest["recovery"]["retained_processes"][0] = {
            key: value
            for key, value in (
                manifest["recovery"]["retained_processes"][0].items()
            )
            if key not in authority_keys
        }
        merge = mock.Mock()
        with self.assertRaisesRegex(
            ramdisk.RamdiskError,
            "lacks durable discovery authority",
        ):
            lifecycle_support._reconcile_unpublished_processes(
                manifest,
                group_alive=lambda ignored: False,
                discover_managed_launches=mock.Mock(return_value=[]),
                merge_usage=merge,
                save_manifest=lambda ignored: None,
            )
        merge.assert_not_called()
        self.assertEqual(state_support._usage_read(canonical), {"0:1": 10})
        self.assertEqual(
            len(manifest["recovery"]["retained_processes"]), 1
        )

    def test_unpublished_recovery_refuses_when_nonce_descendant_alive(self):
        # Original group gone BUT a nonce-attributed descendant survives in a
        # new session: absence is unproven, so no merge and the record is kept.
        merge_id = "7" * 32
        manifest, canonical, entry = self._unpublished_usage_manifest(merge_id)
        escaped = {
            "pid": 4242,
            "uid": entry["uid"],
            "starttime": 999,
            "nonce": entry["nonce"],
            "pgid": 4242,
            "sid": 4242,
            "cmdline": entry["expected_command"],
            "state_dir": entry["state_dir"],
            "weights_dir": entry["weights_dir"],
        }
        merge = mock.Mock()
        with self.assertRaisesRegex(
            ramdisk.RamdiskError,
            "nonce-attributable live process",
        ):
            lifecycle_support._reconcile_unpublished_processes(
                manifest,
                group_alive=lambda ignored: False,
                discover_managed_launches=mock.Mock(return_value=[escaped]),
                merge_usage=merge,
                save_manifest=lambda ignored: None,
            )
        merge.assert_not_called()
        self.assertEqual(state_support._usage_read(canonical), {"0:1": 10})
        self.assertEqual(
            len(manifest["recovery"]["retained_processes"]), 1
        )

    def test_unpublished_recovery_refuses_when_global_discovery_fails(self):
        merge_id = "8" * 32
        manifest, canonical, _entry = self._unpublished_usage_manifest(merge_id)
        merge = mock.Mock()
        with self.assertRaisesRegex(
            ramdisk.RamdiskError,
            "global nonce attribution failed",
        ):
            lifecycle_support._reconcile_unpublished_processes(
                manifest,
                group_alive=lambda ignored: False,
                discover_managed_launches=mock.Mock(
                    side_effect=ramdisk.RamdiskError(
                        "process table changed during scan"
                    )
                ),
                merge_usage=merge,
                save_manifest=lambda ignored: None,
            )
        merge.assert_not_called()
        self.assertEqual(state_support._usage_read(canonical), {"0:1": 10})

    def test_unpublished_recovery_revalidates_immediately_before_merge(self):
        # Preflight proves absence, but a nonce-attributed process appears
        # between preflight and the per-entry merge: the immediate re-check
        # must refuse and mutate no accounting.
        merge_id = "9" * 32
        manifest, canonical, entry = self._unpublished_usage_manifest(merge_id)
        escaped = {
            "pid": 5353,
            "uid": entry["uid"],
            "starttime": 999,
            "nonce": entry["nonce"],
            "pgid": 5353,
            "sid": 5353,
            "cmdline": entry["expected_command"],
            "state_dir": entry["state_dir"],
            "weights_dir": entry["weights_dir"],
        }
        # First call (preflight) sees nothing; the per-entry revalidation sees
        # the escaped descendant and must block the merge.
        discover = mock.Mock(side_effect=[[], [escaped]])
        merge = mock.Mock()
        with self.assertRaisesRegex(
            ramdisk.RamdiskError,
            "nonce-attributable live process",
        ):
            lifecycle_support._reconcile_unpublished_processes(
                manifest,
                group_alive=lambda ignored: False,
                discover_managed_launches=discover,
                merge_usage=merge,
                save_manifest=lambda ignored: None,
            )
        merge.assert_not_called()
        self.assertEqual(state_support._usage_read(canonical), {"0:1": 10})
        self.assertEqual(
            len(manifest["recovery"]["retained_processes"]), 1
        )

    def test_unpublished_recovery_merges_once_when_globally_absent(self):
        merge_id = "1" * 32
        manifest, canonical, _entry = self._unpublished_usage_manifest(merge_id)
        lifecycle_support._reconcile_unpublished_processes(
            manifest,
            group_alive=lambda ignored: False,
            discover_managed_launches=mock.Mock(return_value=[]),
            merge_usage=self._unpublished_merge(canonical),
            save_manifest=lambda ignored: None,
        )
        self.assertEqual(state_support._usage_read(canonical), {"0:1": 12})
        self.assertEqual(
            manifest["recovery"]["retained_processes"], []
        )

    @requires_linux_operational
    def test_unpublished_recovery_real_setsid_descendant_refuses(self):
        # Real kernel reproduction: a managed leader forks a child that calls
        # setsid() and survives in a new session while the original leader's
        # process group disappears. Recovery must not mistake the empty
        # original group for absence while that descendant is alive.
        import subprocess
        import time as _time

        def stat_fields(pid):
            with open("/proc/%d/stat" % pid) as handle:
                raw = handle.read()
            tail = raw[raw.rfind(")") + 2:].split()
            return {
                "state": tail[0],
                "pgid": int(tail[2]),
                "sid": int(tail[3]),
                "starttime": int(tail[19]),
            }

        def cmdline_of(pid):
            with open("/proc/%d/cmdline" % pid, "rb") as handle:
                raw = handle.read()
            return [
                token.decode()
                for token in raw.split(b"\0")
                if token
            ]

        nonce = os.urandom(24).hex()
        state_dir = os.path.join(self.root, "setsid-state")
        weights_dir = os.path.join(self.root, "setsid-weights")
        os.makedirs(state_dir)
        os.makedirs(weights_dir)
        child_pid_file = os.path.join(self.root, "escaped-child.pid")
        leader_script = (
            "import os, time\n"
            "child = os.fork()\n"
            "if child == 0:\n"
            "    os.setsid()\n"
            "    time.sleep(30)\n"
            "    os._exit(0)\n"
            "else:\n"
            "    open(os.environ['CHILD_PID_FILE'], 'w').write(str(child))\n"
            "    os._exit(0)\n"
        )
        expected_command = [sys.executable, "-c", leader_script]
        env = dict(os.environ)
        env.update(
            COLI_MANAGED_NONCE=nonce,
            COLI_STATE_DIR=state_dir,
            COLI_WEIGHTS_DIR=weights_dir,
            CHILD_PID_FILE=child_pid_file,
        )
        leader = subprocess.Popen(
            expected_command,
            env=env,
            start_new_session=True,
        )
        original_pgid = leader.pid
        leader.wait()
        escaped_pid = None
        for _ in range(200):
            if os.path.exists(child_pid_file):
                with open(child_pid_file) as handle:
                    escaped_pid = int(handle.read().strip())
                break
            _time.sleep(0.05)
        self.assertIsNotNone(escaped_pid)
        _time.sleep(0.3)
        try:
            escaped_identity = stat_fields(escaped_pid)
            self.assertNotEqual(escaped_identity["pgid"], original_pgid)
            self.assertEqual(
                escaped_identity["pgid"], escaped_identity["sid"]
            )
            # Mechanism: the original group is gone, yet the global nonce scan
            # still finds the re-sessioned descendant.
            self.assertFalse(
                linux_ops._process_group_alive(original_pgid)
            )
            launcher_cmdline = cmdline_of(os.getpid())
            launcher_starttime = stat_fields(os.getpid())["starttime"]
            scan = None
            for _ in range(20):
                try:
                    scan = linux_ops._managed_launch_processes(
                        nonce,
                        host_uid(),
                        state_dir=state_dir,
                        weights_dir=weights_dir,
                        not_before_starttime=escaped_identity["starttime"],
                        launcher_pid=os.getpid(),
                        launcher_starttime=launcher_starttime,
                        launcher_cmdline=launcher_cmdline,
                        expected_command=expected_command,
                    )
                    break
                except ramdisk.RamdiskError:
                    _time.sleep(0.2)
            self.assertIsNotNone(scan)
            self.assertTrue(
                any(
                    candidate.get("pid") == escaped_pid
                    for candidate in scan
                )
            )

            # Reconciliation with the real scan and group check must refuse to
            # merge accounting while the escaped descendant is alive.
            merge_id = "b" * 32
            model_dir = os.path.join(self.root, "setsid-model")
            os.makedirs(model_dir)
            canonical = os.path.join(model_dir, ".coli_usage")
            state_support._usage_write(canonical, {"0:1": 10})
            state_support._usage_write(
                os.path.join(state_dir, ".coli_usage"),
                {"0:1": 13},
            )
            entry = {
                "pid": original_pgid,
                "pgid": original_pgid,
                "node": None,
                "state_dir": state_dir,
                "usage_baseline": {"0:1": 10},
                "usage_merge_id": merge_id,
                "error": "group absence unproven",
                "authority_version": (
                    lifecycle_support._RETAINED_AUTHORITY_VERSION
                ),
                "nonce": nonce,
                "uid": host_uid(),
                "weights_dir": weights_dir,
                "launch_not_before": escaped_identity["starttime"],
                "launcher_pid": os.getpid(),
                "launcher_starttime": launcher_starttime,
                "launcher_cmdline": launcher_cmdline,
                "expected_command": expected_command,
            }
            manifest = {
                "state": "error",
                "plan": {"model": {"path": model_dir}, "mounts": []},
                "recovery": {
                    "operation": "start",
                    "state": "attention-required",
                    "retained_processes": [entry],
                },
            }
            merge = mock.Mock()
            with self.assertRaisesRegex(
                ramdisk.RamdiskError,
                "incomplete",
            ):
                lifecycle_support._reconcile_unpublished_processes(
                    manifest,
                    group_alive=linux_ops._process_group_alive,
                    discover_managed_launches=(
                        linux_ops._managed_launch_processes
                    ),
                    merge_usage=merge,
                    save_manifest=lambda ignored: None,
                )
            merge.assert_not_called()
            self.assertEqual(
                state_support._usage_read(canonical), {"0:1": 10}
            )
            self.assertEqual(
                len(manifest["recovery"]["retained_processes"]), 1
            )
        finally:
            try:
                os.kill(escaped_pid, 9)
            except (ProcessLookupError, OSError):
                pass
            try:
                os.waitpid(escaped_pid, 0)
            except (ChildProcessError, OSError):
                pass

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
        os.makedirs(manifest["processes"][0]["state_dir"], exist_ok=True)
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
        ops = mock.Mock()
        ops.signal_verified_process_group.side_effect = (
            {"status": "signaled", "members": [12345], "signaled": [12345]},
            {
                "status": "foreign",
                "reason": "reused-pid",
                "members": [12345],
            },
        )
        with mock.patch.object(
            process_support,
            "get_platform_ops",
            return_value=ops,
        ), mock.patch.object(os, "killpg", create=True) as kill:
            failure = ramdisk._terminate_verified_group(
                record,
                term_seconds=0,
                kill_seconds=0,
            )

        kill.assert_not_called()
        self.assertEqual(
            [call.args[1] for call in ops.signal_verified_process_group.call_args_list],
            [signal.SIGTERM, signal.SIGKILL],
        )
        self.assertIn("reused-pid", failure)

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
        process.poll.return_value = 0
        ops = mock.Mock()
        ops.signal_verified_process_group.side_effect = (
            {"status": "signaled", "members": [12346], "signaled": [12346]},
            {"status": "absent", "members": []},
        )
        ramdisk._track_managed_child(process)
        self.addCleanup(ramdisk._forget_managed_child, process.pid)
        with mock.patch.object(
            process_support,
            "get_platform_ops",
            return_value=ops,
        ), mock.patch.object(os, "killpg", create=True) as kill:
            failure = ramdisk._terminate_verified_group(
                record,
                term_seconds=0,
                kill_seconds=0,
            )

        self.assertIsNone(failure)
        kill.assert_not_called()
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

        ops = mock.Mock()
        ops.signal_verified_process_group.return_value = {
            "status": "absent",
            "members": [],
        }
        with mock.patch.object(
            process_support,
            "get_platform_ops",
            return_value=ops,
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

        ops = mock.Mock()
        ops.signal_verified_process_group.return_value = {
            "status": "absent",
            "members": [],
        }
        with mock.patch.object(
            process_support,
            "get_platform_ops",
            return_value=ops,
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
        os.makedirs(manifest["processes"][0]["state_dir"], exist_ok=True)
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
        merge_ids = []

        def replay_merge(record, canonical_usage, *, plan):
            del canonical_usage, plan
            merge_ids.append(record["usage_merge_id"])

        merge = mock.Mock(side_effect=replay_merge)

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
            "bind_usage_transaction": (
                lambda record, plan, reserved_ids: record.setdefault(
                    "usage_merge_id",
                    "e" * 32,
                )
            ),
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

        self.assertEqual(merge.call_count, 2)
        self.assertEqual(len(set(merge_ids)), 1)
        self.assertEqual(stopped["state"], "stopped")
        self.assertIn("usage_merged_at", stopped["processes"][0])
        self.assertNotIn("usage_merge_error", stopped["processes"][0])

    def test_stop_persists_legacy_journal_id_before_usage_replay(self):
        manifest = self.manifest(
            state="running",
            processes=[{"pid": 12361}],
        )
        record = manifest["processes"][0]
        record.pop("usage_merge_id", None)
        state_dir = record["state_dir"]
        os.makedirs(state_dir, exist_ok=True)
        merge_id = "a" * 32
        state_support._atomic_json(
            os.path.join(state_dir, ".coli_usage.delta.json"),
            {"version": 1, "id": merge_id, "delta": {"0:1": 2}},
        )
        events = []

        def save(current):
            events.append(
                ("save", current["processes"][0].get("usage_merge_id"))
            )

        def merge(current, canonical, *, plan):
            del canonical, plan
            events.append(("merge", current.get("usage_merge_id")))

        with mock.patch.object(
            ramdisk,
            "_load_manifest",
            return_value=manifest,
        ), mock.patch.object(
            ramdisk,
            "_managed_launch_processes",
            return_value=[],
        ), mock.patch.object(
            ramdisk,
            "_process_matches",
            return_value=(False, "not-running", None),
        ), mock.patch.object(
            ramdisk,
            "_managed_child_liveness",
            return_value=False,
        ), mock.patch.object(
            ramdisk,
            "_group_alive",
            return_value=False,
        ), mock.patch.object(
            ramdisk,
            "_save_manifest",
            side_effect=save,
        ), mock.patch.object(
            ramdisk,
            "_merge_usage",
            side_effect=merge,
        ):
            ramdisk.stop.__wrapped__()

        self.assertEqual(events[0], ("save", merge_id))
        self.assertIn(("merge", merge_id), events)

    def test_start_persists_legacy_journal_id_before_usage_replay(self):
        manifest = self.manifest(
            state="stopped",
            processes=[
                {
                    "pid": 12362,
                    "stopped_at": "2026-08-01T00:00:00Z",
                }
            ],
        )
        record = manifest["processes"][0]
        record.pop("usage_merge_id", None)
        state_dir = record["state_dir"]
        os.makedirs(state_dir, exist_ok=True)
        merge_id = "b" * 32
        state_support._atomic_json(
            os.path.join(state_dir, ".coli_usage.delta.json"),
            {"version": 1, "id": merge_id, "delta": {"0:1": 2}},
        )
        events = []

        def save(current):
            events.append(
                ("save", current["processes"][0].get("usage_merge_id"))
            )

        def merge(current, canonical, *, plan):
            del canonical, plan
            events.append(("merge", current.get("usage_merge_id")))

        with mock.patch.object(
            ramdisk,
            "_load_manifest",
            return_value=manifest,
        ), mock.patch.object(
            ramdisk,
            "_assert_effective_masks_unchanged",
        ), mock.patch.object(
            ramdisk,
            "_assert_ready_mounts",
        ), mock.patch.object(
            ramdisk,
            "_process_matches",
            return_value=(False, "not-running", None),
        ), mock.patch.object(
            ramdisk,
            "_managed_child_liveness",
            return_value=False,
        ), mock.patch.object(
            ramdisk,
            "_save_manifest",
            side_effect=save,
        ), mock.patch.object(
            ramdisk,
            "_merge_usage",
            side_effect=merge,
        ), mock.patch.object(
            ramdisk,
            "_persisted_base_port",
            side_effect=ramdisk.RamdiskError("stop after recovery"),
        ):
            with self.assertRaisesRegex(
                ramdisk.RamdiskError,
                "stop after recovery",
            ):
                ramdisk.start.__wrapped__(
                    argparse.Namespace(base_port=None),
                    cli_path=sys.executable,
                )

        self.assertEqual(events[0], ("save", merge_id))
        self.assertEqual(events[1], ("merge", merge_id))

    def test_stop_binds_all_transactions_before_save_or_signal(self):
        manifest = self.manifest(
            state="running",
            mount_paths=[
                "/mnt/colibri-test/node0",
                "/mnt/colibri-test/node1",
            ],
            processes=[{"pid": 12371}, {"pid": 12372}],
        )
        events = []
        terminate = mock.Mock()
        merge = mock.Mock()

        def bind(record, plan, reserved_ids):
            del plan
            merge_id = ("a" if record["pid"] == 12371 else "b") * 32
            self.assertNotIn(merge_id, reserved_ids)
            events.append(("bind", record["pid"], merge_id))
            record["usage_merge_id"] = merge_id
            return merge_id

        def fail_save(current):
            events.append(
                (
                    "save",
                    tuple(
                        record.get("usage_merge_id")
                        for record in current["processes"]
                    ),
                )
            )
            raise OSError("transaction authority save failed")

        with self.assertRaisesRegex(
            OSError,
            "transaction authority save failed",
        ):
            lifecycle_support.stop(
                load_manifest=lambda required=True: manifest,
                process_matches=lambda record: (True, "running", {}),
                group_alive=lambda pgid: True,
                managed_child_liveness=lambda pid: False,
                save_manifest=fail_save,
                terminate_verified_group=terminate,
                merge_usage=merge,
                bind_usage_transaction=bind,
            )

        self.assertEqual(
            [event[0] for event in events],
            ["bind", "bind", "save"],
        )
        terminate.assert_not_called()
        merge.assert_not_called()

    def test_duplicate_legacy_journals_fail_before_record_mutation(self):
        manifest = self.manifest(
            state="running",
            mount_paths=[
                "/mnt/colibri-test/node0",
                "/mnt/colibri-test/node1",
            ],
            processes=[{"pid": 12373}, {"pid": 12374}],
        )
        merge_id = "c" * 32
        for record in manifest["processes"]:
            os.makedirs(record["state_dir"], exist_ok=True)
            state_support._atomic_json(
                os.path.join(record["state_dir"], ".coli_usage.delta.json"),
                {
                    "version": 1,
                    "id": merge_id,
                    "delta": {"0:1": 1},
                },
            )
        save = mock.Mock()
        terminate = mock.Mock()
        merge = mock.Mock()

        with self.assertRaisesRegex(
            ramdisk.RamdiskError,
            "duplicate usage transaction",
        ):
            lifecycle_support.stop(
                load_manifest=lambda required=True: manifest,
                process_matches=lambda record: (False, "not-running", None),
                group_alive=lambda pgid: False,
                managed_child_liveness=lambda pid: False,
                save_manifest=save,
                terminate_verified_group=terminate,
                merge_usage=merge,
                bind_usage_transaction=ramdisk._bind_usage_transaction,
            )

        self.assertTrue(
            all(
                record.get("usage_merge_id") is None
                for record in manifest["processes"]
            )
        )
        save.assert_not_called()
        terminate.assert_not_called()
        merge.assert_not_called()

    def test_start_duplicate_live_orphan_journals_fail_before_replay(self):
        manifest = self.manifest(
            state="stopped",
            mount_paths=[
                "/mnt/colibri-test/node0",
                "/mnt/colibri-test/node1",
            ],
        )
        model_dir = manifest["plan"]["model"]["path"]
        canonical = os.path.join(model_dir, ".coli_usage")
        os.makedirs(model_dir)
        ramdisk._usage_write(canonical, {"0:1": 10})
        canonical_before = Path(canonical).read_bytes()
        merge_id = "7" * 32
        journals = []
        journal_bytes = []
        for node, delta in enumerate((2, 5)):
            state_dir = self.recovery_state_dir(node=node)
            os.makedirs(state_dir, mode=0o700)
            journal = os.path.join(state_dir, ".coli_usage.delta.json")
            state_support._atomic_json(
                journal,
                {"version": 1, "id": merge_id, "delta": {"0:1": delta}},
            )
            journals.append(journal)
            journal_bytes.append(Path(journal).read_bytes())

        with mock.patch.object(
            ramdisk,
            "_load_manifest",
            return_value=manifest,
        ), mock.patch.object(
            ramdisk,
            "_assert_effective_masks_unchanged",
        ), mock.patch.object(
            ramdisk,
            "_assert_ready_mounts",
        ), mock.patch.object(
            ramdisk,
            "_save_manifest",
        ) as save, mock.patch.object(
            ramdisk,
            "_persisted_base_port",
            side_effect=ramdisk.RamdiskError("continued past journal preflight"),
        ):
            with self.assertRaisesRegex(
                ramdisk.RamdiskError,
                "duplicate usage transaction",
            ):
                ramdisk.start.__wrapped__(
                    argparse.Namespace(base_port=None),
                    cli_path=sys.executable,
                )

        save.assert_not_called()
        self.assertEqual(Path(canonical).read_bytes(), canonical_before)
        self.assertEqual(
            [Path(path).read_bytes() for path in journals],
            journal_bytes,
        )

    def test_start_orphan_journal_cannot_reuse_manifest_authority(self):
        merge_id = "6" * 32
        manifest = self.manifest(
            state="stopped",
            mount_paths=[
                "/mnt/colibri-test/node0",
                "/mnt/colibri-test/node1",
            ],
            processes=[
                {
                    "pid": 12375,
                    "stopped_at": "2026-08-01T00:00:00Z",
                    "usage_merge_id": merge_id,
                }
            ],
        )
        model_dir = manifest["plan"]["model"]["path"]
        canonical = os.path.join(model_dir, ".coli_usage")
        os.makedirs(model_dir)
        ramdisk._usage_write(canonical, {"0:1": 10})
        canonical_before = Path(canonical).read_bytes()
        os.makedirs(self.recovery_state_dir(node=0), mode=0o700)
        orphan_state = self.recovery_state_dir(node=1)
        os.makedirs(orphan_state, mode=0o700)
        journal = os.path.join(orphan_state, ".coli_usage.delta.json")
        state_support._atomic_json(
            journal,
            {"version": 1, "id": merge_id, "delta": {"0:1": 5}},
        )
        journal_before = Path(journal).read_bytes()

        with mock.patch.object(
            ramdisk,
            "_load_manifest",
            return_value=manifest,
        ), mock.patch.object(
            ramdisk,
            "_assert_effective_masks_unchanged",
        ), mock.patch.object(
            ramdisk,
            "_assert_ready_mounts",
        ), mock.patch.object(
            ramdisk,
            "_process_matches",
            return_value=(False, "not-running", None),
        ), mock.patch.object(
            ramdisk,
            "_managed_child_liveness",
            return_value=False,
        ), mock.patch.object(
            ramdisk,
            "_save_manifest",
        ) as save:
            with self.assertRaisesRegex(
                ramdisk.RamdiskError,
                "duplicate usage transaction",
            ):
                ramdisk.start.__wrapped__(
                    argparse.Namespace(base_port=None),
                    cli_path=sys.executable,
                )

        save.assert_not_called()
        self.assertEqual(Path(canonical).read_bytes(), canonical_before)
        self.assertEqual(Path(journal).read_bytes(), journal_before)

    def test_start_mint_reserves_preflighted_orphan_journal_id(self):
        manifest = self.manifest(state="stopped")
        model_dir = manifest["plan"]["model"]["path"]
        os.makedirs(model_dir)
        ramdisk._usage_write(
            os.path.join(model_dir, ".coli_usage"),
            {"0:1": 10},
        )
        state_dir = self.recovery_state_dir()
        os.makedirs(state_dir, mode=0o700)
        orphan_id = "8" * 32
        fresh_id = "9" * 32
        state_support._atomic_json(
            os.path.join(state_dir, ".coli_usage.delta.json"),
            {"version": 1, "id": orphan_id, "delta": {"0:1": 2}},
        )
        mint_attempts = []

        def token_hex(size):
            if size == 24:
                return "a" * 48
            self.assertEqual(size, 16)
            mint_attempts.append(len(mint_attempts))
            return orphan_id if len(mint_attempts) == 1 else fresh_id

        class FakeSocket:
            def bind(self, address):
                del address

            def close(self):
                pass

        with mock.patch.object(
            ramdisk,
            "_load_manifest",
            return_value=manifest,
        ), mock.patch.object(
            ramdisk,
            "_assert_effective_masks_unchanged",
        ), mock.patch.object(
            ramdisk,
            "_assert_ready_mounts",
        ), mock.patch.object(
            ramdisk,
            "_save_manifest",
        ), mock.patch.object(
            ramdisk,
            "_recover_delta",
        ), mock.patch.object(
            ramdisk,
            "_usage_read",
            return_value={"0:1": 10},
        ), mock.patch.object(
            ramdisk,
            "_usage_write",
        ), mock.patch.object(
            ramdisk,
            "_admit_concurrent_runtimes",
        ), mock.patch.object(
            ramdisk,
            "_current_process_identity",
            return_value={
                "pid": 700,
                "uid": host_uid(),
                "starttime": 90,
                "cmdline": ["coli", "ramdisk", "start"],
            },
        ), mock.patch.object(
            ramdisk,
            "_process_start_boundary",
            side_effect=ramdisk.RamdiskError("stop after transaction mint"),
        ), mock.patch.object(
            ramdisk.socket,
            "socket",
            side_effect=lambda *args, **kwargs: FakeSocket(),
        ), mock.patch.object(
            lifecycle_support.secrets,
            "token_hex",
            side_effect=token_hex,
        ):
            with self.assertRaisesRegex(
                ramdisk.RamdiskError,
                "stop after transaction mint",
            ):
                ramdisk.start.__wrapped__(
                    argparse.Namespace(base_port=None),
                    cli_path=sys.executable,
                )

        self.assertEqual(len(mint_attempts), 2)

    def test_expected_orphan_journal_disappearance_fails_closed(self):
        model_dir = os.path.join(self.root, "model")
        canonical = os.path.join(model_dir, ".coli_usage")
        state_dir = os.path.join(self.root, "orphan-state")
        os.makedirs(model_dir)
        os.makedirs(state_dir)
        ramdisk._usage_write(canonical, {"0:1": 10})

        with self.assertRaisesRegex(
            ramdisk.RamdiskError,
            "expected usage delta journal is absent",
        ):
            ramdisk._recover_delta(
                state_dir,
                canonical,
                expected_merge_id="a" * 32,
            )

        self.assertEqual(ramdisk._usage_read(canonical), {"0:1": 10})

    def test_usage_transaction_mint_skips_reserved_collision(self):
        reserved = {"d" * 32}
        with mock.patch.object(
            lifecycle_support.secrets,
            "token_hex",
            side_effect=("d" * 32, "e" * 32),
        ):
            merge_id = lifecycle_support._mint_usage_transaction_id(reserved)

        self.assertEqual(merge_id, "e" * 32)
        self.assertEqual(reserved, {"d" * 32, "e" * 32})

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
        os.makedirs(manifest["processes"][0]["state_dir"], exist_ok=True)
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

    def test_prepare_save_ambiguity_preserves_strongest_exact_mount(self):
        mount_path = os.path.join(self.root, "ramdisk-mount")
        exact_identity = {
            "mount_id": 41,
            "device": "0:41",
            "filesystem": "tmpfs",
            "source": "tmpfs",
        }
        plan = {
            "blockers": [],
            "mount_root": mount_path,
            "mounts": [
                {
                    "path": mount_path,
                    "node": None,
                    "size_bytes": 4096,
                }
            ],
            "topology": "interleaved",
            "model": {
                "path": os.path.join(self.root, "model"),
                "fingerprint": self.FINGERPRINT,
            },
            "hardware": {
                "swap": {"used_bytes": 0},
            },
            "mount_options": {},
        }
        durable = {"manifest": None}
        save_calls = {"count": 0}

        def ambiguous_save(current):
            save_calls["count"] += 1
            durable["manifest"] = copy.deepcopy(current)
            if save_calls["count"] in (3, 4):
                raise OSError("post-replace directory fsync failed")

        observed = iter((None, copy.deepcopy(exact_identity)))
        unmount = mock.Mock()
        with self.assertRaisesRegex(
            ramdisk.RamdiskError,
            "post-replace directory fsync failed",
        ):
            lifecycle_support.prepare(
                argparse.Namespace(base_port=8000, yes=True),
                display_plan=False,
                load_manifest=lambda required=False: None,
                build_plan=lambda args: copy.deepcopy(plan),
                managed_ports_for_plan=lambda current, base: [base],
                plan_confirmation_token=lambda current: "token",
                render_plan=mock.Mock(),
                confirm=mock.Mock(),
                save_manifest=ambiguous_save,
                mount_at=lambda path: next(observed),
                mount_tmpfs=mock.Mock(),
                umount_path=unmount,
                validate_mount=mock.Mock(return_value=exact_identity),
                populate_mount=mock.Mock(),
                validate_namespace=mock.Mock(),
                source_still_matches=mock.Mock(),
                ensure_busy_mount_scan_available=mock.Mock(),
                durable_unlink=mock.Mock(),
                manifest_path=lambda: os.path.join(self.root, "manifest.json"),
                mount_table=mock.Mock(return_value=[]),
                path_is_below=lambda path, parent: False,
                busy_mount_references=mock.Mock(return_value=[]),
            )

        persisted = durable["manifest"]["mounts"][0]
        self.assertEqual(persisted["ownership"], "identified")
        self.assertEqual(persisted["identity"], exact_identity)
        self.assertEqual(persisted["cleanup"]["state"], "retained")
        unmount.assert_not_called()

    def test_prepare_recovery_candidate_unions_durable_pending_records(self):
        first_path = os.path.join(self.root, "mount-a")
        second_path = os.path.join(self.root, "mount-b")
        durable = {
            "state": "preparing",
            "mounts": [
                {
                    "path": first_path,
                    "operation_id": "deploy:mount:0",
                    "ownership": "pending",
                },
                {
                    "path": second_path,
                    "operation_id": "deploy:mount:1",
                    "ownership": "pending",
                },
            ],
        }
        current = {
            "state": "preparing",
            "mounts": [
                {
                    "path": first_path,
                    "operation_id": "deploy:mount:0",
                    "ownership": "identified",
                    "identity": {"mount_id": 42, "device": "0:42"},
                }
            ],
        }

        recovery = lifecycle_support._strongest_prepare_recovery_manifest(
            current,
            durable,
        )

        by_path = {record["path"]: record for record in recovery["mounts"]}
        self.assertEqual(by_path[first_path]["ownership"], "identified")
        self.assertEqual(by_path[first_path]["identity"]["mount_id"], 42)
        self.assertEqual(by_path[second_path]["ownership"], "pending")

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
            "state_dir": "/state/node-0",
            "weights_dir": "/mnt/weights",
        }
        members = [
            {
                "pid": 101,
                "uid": host_uid(),
                "inert": False,
                "nonce": "a" * 48,
                "pgid": 101,
                "sid": 101,
                "state_dir": "/state/node-0",
                "weights_dir": "/mnt/weights",
            },
            {
                "pid": 102,
                "uid": host_uid(),
                "inert": False,
                "nonce": "a" * 48,
                "pgid": 101,
                "sid": 101,
                "state_dir": "/state/node-0",
                "weights_dir": "/mnt/weights",
            },
            {
                "pid": 103,
                "uid": host_uid(),
                "inert": True,
                "starttime": 203,
                "nonce": None,
                "pgid": 101,
                "sid": 101,
                "state_dir": None,
                "weights_dir": None,
            },
        ]

        def proc_text(path, default=""):
            if path == "/proc/101/status":
                return "VmRSS:\t100 kB\n"
            if path == "/proc/102/status":
                return "VmRSS:\t900 kB\n"
            if path == "/proc/103/status":
                self.fail("inert zombie must be excluded from RSS metrics")
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

    def test_status_distinguishes_absent_mount_from_unknown_observation(self):
        manifest = self.manifest(state="ready")
        with mock.patch.object(
            ramdisk, "_load_manifest", return_value=manifest
        ), mock.patch.object(ramdisk, "_mount_at", return_value=None):
            absent = ramdisk.status(deep=False)["mounts"][0]

        with mock.patch.object(
            ramdisk, "_load_manifest", return_value=manifest
        ), mock.patch.object(
            ramdisk,
            "_mount_at",
            side_effect=OSError("mount table unreadable"),
        ):
            unknown = ramdisk.status(deep=False)["mounts"][0]

        self.assertIs(absent["mounted"], False)
        self.assertIsNone(absent["option_error"])
        self.assertIsNone(unknown["mounted"])
        self.assertIn("mount table unreadable", unknown["option_error"])
        self.assertFalse(unknown["verified"])

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
                "uid": host_uid(),
                "port": 8000,
                "node": None,
                "state_dir": os.path.join(self.root, "pending-state"),
                "weights_dir": "/mnt/colibri-test",
                "launch_not_before": 100,
                "launcher_pid": 700,
                "launcher_starttime": 90,
                "launcher_cmdline": ["coli", "ramdisk", "start"],
                "expected_command": ["coli", "serve"],
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
        self.assertNotIn("launch_not_before", serialized)
        self.assertNotIn("launcher_pid", serialized)
        self.assertNotIn("launcher_starttime", serialized)
        self.assertNotIn("launcher_cmdline", serialized)
        self.assertNotIn("expected_command", serialized)
        self.assertNotIn("weights_dir", serialized)

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
                    "usage_merge_id": "d" * 32,
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
