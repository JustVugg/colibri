import json
import os
import tempfile
import unittest
from pathlib import Path

from tools.qpack_install_policy import (
    JOURNAL,
    MANIFEST,
    MAX_FILE_SIZE,
    QpackFile,
    QpackInstallError,
    begin_install,
    commit_file,
    parse_manifest,
    publish_manifest,
    resume_decision,
    validate_delivery,
)


class QpackInstallPolicyTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "model.qpack"
        self.raw = self.manifest({
            "model.safetensors": 7,
            "packed_experts/layout.json": 5,
        })
        self.plan = parse_manifest(self.raw, "hf:owner/model@0123456789abcdef")
        self.sessions = []
        self.addCleanup(self.close_sessions)

    def close_sessions(self):
        for session in reversed(self.sessions):
            session.close()

    def begin(self, plan=None):
        session = begin_install(self.root, plan or self.plan)
        self.sessions.append(session)
        return session

    @staticmethod
    def manifest(files, **extra):
        value = {
            "magic": "QPACK",
            "version": 1,
            "modelName": "tiny-qwen",
            "sourceCheckpoint": "owner/model@0123456789abcdef",
            "quantBits": 4,
            "quantGroupSize": 64,
            "files": files,
        }
        value.update(extra)
        return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()

    def file(self, name):
        return next(file for file in self.plan.files if file.name == name)

    def write_partial(self, session, file, data):
        decision = resume_decision(session, file)
        decision.partial_path.write_bytes(data)
        return decision.partial_path

    def commit_all(self, session):
        for file in self.plan.files:
            self.write_partial(session, self.file(file.name),
                               bytes([file.size]) * file.size)
            commit_file(session, file)

    def test_manifest_is_sorted_and_bound_to_exact_bytes_and_source(self):
        reordered = self.raw + b"\n"
        other = parse_manifest(reordered, self.plan.source_id)
        self.assertEqual([file.name for file in other.files],
                         ["model.safetensors", "packed_experts/layout.json"])
        self.assertNotEqual(other.manifest_sha256, self.plan.manifest_sha256)
        self.assertNotEqual(parse_manifest(self.raw, "local:fixture"), self.plan)

    def test_manifest_rejects_duplicate_keys(self):
        raw = (b'{"magic":"QPACK","version":1,"files":'
               b'{"packed_experts/layout.json":5,'
               b'"packed_experts/layout.json":6}}')
        with self.assertRaisesRegex(QpackInstallError, "duplicate JSON key"):
            parse_manifest(raw, "local:fixture")

    def test_manifest_rejects_nonfinite_numbers_and_escaped_nuls(self):
        raw = self.raw.replace(b'"quantBits":4', b'"quantBits":NaN')
        with self.assertRaisesRegex(QpackInstallError, "non-finite"):
            parse_manifest(raw, "local:fixture")
        value = json.loads(self.raw)
        value["modelName"] = "tiny\x00qwen"
        with self.assertRaisesRegex(QpackInstallError, "embedded NUL"):
            parse_manifest(json.dumps(value).encode(), "local:fixture")

    def test_manifest_rejects_unsafe_and_reserved_paths(self):
        invalid = [
            "../escape", "/absolute", "a//b", "a/./b", "a/../b",
            "a\\b", "manifest.json", ".qpack-install.json",
            ".qpack-install.lock", ".manifest.json.part", "nul\x00name",
            "C:/escape", "C:escape", "packed_experts/CON",
            "packed_experts/aux.txt", "packed_experts/trailing. ",
            "packed_experts/file:stream", "packed_experts/a?.bin",
            "packed_experts/caf\u00e9.bin",
        ]
        for name in invalid:
            files = {"packed_experts/layout.json": 5, name: 1}
            with self.subTest(name=name), self.assertRaises(QpackInstallError):
                parse_manifest(self.manifest(files), "local:fixture")

    def test_manifest_rejects_invalid_sizes_and_layout_is_not_required(self):
        for size in (True, False, 0, -1, 1.5, MAX_FILE_SIZE + 1):
            with self.subTest(size=size), self.assertRaises(QpackInstallError):
                parse_manifest(self.manifest({
                    "packed_experts/layout.json": size,
                }), "local:fixture")
        # The production Qwen3-Next-80B manifest declares no layout.json;
        # Swiftlet's reader never consults the files table for it.
        plan = parse_manifest(
            self.manifest({"model.safetensors": 7}), "local:fixture")
        self.assertEqual(plan.files, (QpackFile("model.safetensors", 7),))
        with self.assertRaisesRegex(QpackInstallError, "source_id"):
            parse_manifest(self.raw, "")

    def test_real_swiftlet_manifests_are_accepted_with_or_without_layout(self):
        fixtures = Path(__file__).parent / "fixtures"
        # Qwen3-Next-80B, as published on Swiftlet's mirror: 49 entries,
        # model.safetensors plus 48 layer blobs, no layout.json, no aux files.
        plan = parse_manifest(
            (fixtures / "swiftlet_qwen3_next_80b_manifest.json").read_bytes(),
            "mirror:fixture")
        names = [file.name for file in plan.files]
        self.assertEqual(len(names), 49)
        self.assertNotIn("packed_experts/layout.json", names)
        self.assertEqual(
            names,
            ["model.safetensors"]
            + [f"packed_experts/layer_{index:02d}.bin" for index in range(48)])
        self.assertTrue(all(file.size > 0 for file in plan.files))
        # Qwen3.6-35B, same writer, declares layout.json and the tokenizer.
        plan = parse_manifest(
            (fixtures / "swiftlet_qwen36_35b_manifest.json").read_bytes(),
            "mirror:fixture")
        self.assertEqual(len(plan.files), 47)
        self.assertIn(QpackFile("packed_experts/layout.json", 2048), plan.files)

    def test_declared_layout_with_wrong_size_cannot_commit_or_publish(self):
        session = self.begin()
        layout = self.file("packed_experts/layout.json")
        self.write_partial(session, layout, b"x" * (layout.size + 1))
        with self.assertRaisesRegex(QpackInstallError, "not complete"):
            commit_file(session, layout)
        self.write_partial(session, layout, b"x" * (layout.size - 1))
        with self.assertRaisesRegex(QpackInstallError, "not complete"):
            commit_file(session, layout)
        self.write_partial(session, self.file("model.safetensors"), b"w" * 7)
        commit_file(session, self.file("model.safetensors"))
        with self.assertRaisesRegex(QpackInstallError, "before all files"):
            publish_manifest(session, self.raw)
        self.assertFalse((self.root / MANIFEST).exists())

    def test_manifest_matches_swiftlet_required_fields_and_file_limit(self):
        for field in ("modelName", "sourceCheckpoint"):
            value = json.loads(self.raw)
            value.pop(field)
            with self.subTest(field=field), self.assertRaisesRegex(
                    QpackInstallError, "supported QPACK"):
                parse_manifest(json.dumps(value).encode(), "local:fixture")
        for field in ("quantBits", "quantGroupSize"):
            value = json.loads(self.raw)
            value[field] = True
            with self.subTest(field=field), self.assertRaisesRegex(
                    QpackInstallError, field):
                parse_manifest(json.dumps(value).encode(), "local:fixture")
        too_many = {"packed_experts/layout.json": 5}
        too_many.update({f"packed_experts/file-{index}.bin": 1
                         for index in range(8192)})
        with self.assertRaisesRegex(QpackInstallError, "files table"):
            parse_manifest(self.manifest(too_many), "local:fixture")

    def test_manifest_rejects_plan_wide_path_collisions(self):
        collisions = [
            {"packed_experts/layout.json": 5, "foo": 1, "foo.part": 1},
            {"packed_experts/layout.json": 5, "foo": 1, "foo/bar": 1},
            {"packed_experts/layout.json": 5, "Foo": 1, "foo": 1},
        ]
        for files in collisions:
            with self.subTest(files=files), self.assertRaisesRegex(
                    QpackInstallError, "namespace collision"):
                parse_manifest(self.manifest(files), "local:fixture")

    def test_begin_binds_partials_and_rejects_unbound_or_mismatched_work(self):
        self.root.mkdir()
        (self.root / "model.safetensors.part").write_bytes(b"old")
        with self.assertRaisesRegex(QpackInstallError, "unbound"):
            begin_install(self.root, self.plan)
        (self.root / "model.safetensors.part").unlink()

        session = self.begin()
        with self.assertRaisesRegex(QpackInstallError, "locked"):
            begin_install(self.root, self.plan)
        session.close()
        other_raw = self.manifest({
            "model.safetensors": 8,
            "packed_experts/layout.json": 5,
        })
        other = parse_manifest(other_raw, self.plan.source_id)
        with self.assertRaisesRegex(QpackInstallError, "another source or manifest"):
            begin_install(self.root, other)

    def test_malformed_journal_and_closed_session_fail_closed(self):
        session = self.begin()
        session.close()
        (self.root / JOURNAL).write_text('{"schema":"wrong"}', encoding="utf-8")
        with self.assertRaisesRegex(QpackInstallError, "journal is invalid"):
            begin_install(self.root, self.plan)
        with self.assertRaisesRegex(QpackInstallError, "closed or invalid"):
            resume_decision(session, self.plan.files[0])

    def test_resume_decisions_cover_missing_partial_and_committed_files(self):
        session = self.begin()
        file = self.file("model.safetensors")
        decision = resume_decision(session, file)
        self.assertEqual((decision.action, decision.offset), ("restart", 0))

        decision.partial_path.write_bytes(b"abc")
        decision = resume_decision(session, file)
        self.assertEqual((decision.action, decision.offset), ("resume", 3))

        decision.partial_path.write_bytes(b"too-large")
        decision = resume_decision(session, file)
        self.assertEqual((decision.action, decision.offset), ("restart", 0))

        decision.partial_path.write_bytes(b"1234567")
        commit_file(session, file)
        decision = resume_decision(session, file)
        self.assertEqual((decision.action, decision.offset), ("complete", 7))

    def test_relative_root_remains_stable_after_working_directory_change(self):
        original = Path.cwd()
        self.addCleanup(os.chdir, original)
        os.chdir(self.temporary.name)
        session = begin_install(Path("relative") / "model.qpack", self.plan)
        self.sessions.append(session)
        os.chdir(original)
        decision = resume_decision(session, self.plan.files[0])
        self.assertTrue(decision.partial_path.is_absolute())
        self.assertEqual(decision.offset, 0)

    def test_delivery_requires_exact_offsets_bounds_and_final_length(self):
        file = QpackFile("model.safetensors", 10)
        self.assertEqual(validate_delivery(file, 3, 3, 4), 7)
        self.assertEqual(validate_delivery(file, 3, 3, 7, complete=True), 10)
        invalid = [
            (-1, 0, 1, False),
            (3, 0, 7, True),
            (3, 3, 8, False),
            (3, 3, 6, True),
        ]
        for requested, delivered, size, complete in invalid:
            with self.subTest(values=(requested, delivered, size, complete)), \
                    self.assertRaises(QpackInstallError):
                validate_delivery(file, requested, delivered, size,
                                  complete=complete)

    def test_incomplete_commit_never_publishes_a_final_path(self):
        session = self.begin()
        file = self.file("model.safetensors")
        partial = self.write_partial(session, file, b"short")
        with self.assertRaisesRegex(QpackInstallError, "not complete"):
            commit_file(session, file)
        self.assertTrue(partial.exists())
        self.assertFalse(self.root.joinpath(file.name).exists())

    def test_manifest_is_published_last_and_completion_is_idempotent(self):
        session = self.begin()
        first = self.plan.files[0]
        self.write_partial(session, first, bytes([first.size]) * first.size)
        commit_file(session, first)
        with self.assertRaisesRegex(QpackInstallError, "all files"):
            publish_manifest(session, self.raw)
        self.assertFalse((self.root / MANIFEST).exists())

        remaining = self.plan.files[1]
        self.write_partial(session, remaining,
                           bytes([remaining.size]) * remaining.size)
        commit_file(session, remaining)
        publish_manifest(session, self.raw)
        self.assertEqual((self.root / MANIFEST).read_bytes(), self.raw)
        self.assertFalse((self.root / JOURNAL).exists())

        session.close()
        complete_session = self.begin()
        self.assertTrue(complete_session.complete)
        self.assertFalse((self.root / JOURNAL).exists())

    def test_crash_boundaries_remain_recoverable(self):
        session = self.begin()
        first = self.plan.files[0]
        self.write_partial(session, first, bytes([first.size]) * first.size)
        self.assertEqual(resume_decision(session, first).offset, first.size)
        session.close()
        session = self.begin()
        commit_file(session, first)
        self.assertEqual(resume_decision(session, first).action, "complete")

        for file in self.plan.files[1:]:
            self.write_partial(session, file, bytes([file.size]) * file.size)
            commit_file(session, file)
        stale_manifest_temp = self.root / ".manifest.json.part"
        stale_manifest_temp.write_bytes(b"interrupted")
        publish_manifest(session, self.raw)
        self.assertEqual((self.root / MANIFEST).read_bytes(), self.raw)
        self.assertFalse(stale_manifest_temp.exists())

    def test_existing_manifest_cleans_a_stale_matching_journal(self):
        session = self.begin()
        self.commit_all(session)
        (self.root / MANIFEST).write_bytes(self.raw)
        self.assertTrue((self.root / JOURNAL).exists())
        session.close()
        complete_session = self.begin()
        self.assertTrue(complete_session.complete)
        self.assertFalse((self.root / JOURNAL).exists())

    @unittest.skipUnless(hasattr(os, "symlink"), "symlinks unsupported")
    def test_manifest_symlink_cannot_mark_install_complete(self):
        session = self.begin()
        self.commit_all(session)
        external = Path(self.temporary.name) / "external-manifest.json"
        external.write_bytes(self.raw)
        os.symlink(external, self.root / MANIFEST)
        with self.assertRaisesRegex(QpackInstallError, "symlink"):
            publish_manifest(session, self.raw)
        self.assertTrue((self.root / JOURNAL).exists())

    @unittest.skipUnless(hasattr(os, "symlink"), "symlinks unsupported")
    def test_symlinked_artifact_parent_is_rejected(self):
        outside = Path(self.temporary.name) / "outside"
        outside.mkdir()
        self.root.mkdir()
        os.symlink(outside, self.root / "packed_experts", target_is_directory=True)
        with self.assertRaisesRegex(QpackInstallError, "symlink"):
            begin_install(self.root, self.plan)


if __name__ == "__main__":
    unittest.main()
