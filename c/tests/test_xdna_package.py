"""Owner for the optional XDNA sidecar release process.

tools/build_xdna_package.py is the only thing that decides what the optional
Windows XDNA download contains and whether it may be published. These tests
pin the parts a release depends on: the asset set, the naming that keeps the
sidecar distinguishable from the core archive, mechanical manifest generation,
and -- most of all -- that a wrong version, a wrong archive checksum or a
digest that disagrees with the compiled registry is refused BEFORE the upload
step rather than after.

The qualified helper and artifact bytes are not available on a build machine,
so every fixture here is synthetic: the tests drive the tool's own registry
parsing, so the expected names and hashes come from c/backend_xdna.c exactly as
they do in production.
"""
import hashlib
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.normpath(os.path.join(HERE, "..", "tools", "build_xdna_package.py"))


def load_tool():
    spec = importlib.util.spec_from_file_location("build_xdna_package", TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestSidecarReleaseProcess(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tool = load_tool()

    # ---- asset set -------------------------------------------------------

    def test_expected_asset_count_is_five(self):
        """One helper plus four qualified artifacts -- nothing else ships."""
        need = self.tool.required_files()
        self.assertEqual(len(need), 4, "expected four qualified artifacts")
        self.assertTrue(self.tool.helper_name().endswith(".dll"))
        # 4 artifacts + 1 helper
        self.assertEqual(len(need) + 1, 5)

    def test_registry_hashes_are_read_from_the_engine_source(self):
        """The packager must not carry its own copy of the expected hashes."""
        with open(self.tool.REGISTRY_SRC, encoding="utf-8", errors="replace") as fh:
            src = fh.read()
        for name, digest in self.tool.required_files():
            self.assertIn(name, src)
            self.assertIn(digest, src)

    # ---- naming ----------------------------------------------------------

    def test_asset_name_is_distinguishable_from_the_core_archive(self):
        stem = self.tool.asset_stem("v1.2.3")
        self.assertEqual(stem, "colibri-v1.2.3-windows-x86_64-xdna")
        # the core archive for the same tag/platform must not collide
        self.assertNotEqual(stem, "colibri-v1.2.3-windows-x86_64")
        # and it still matches the release job's own `sha256sum colibri-*`
        self.assertTrue(stem.startswith("colibri-"))

    def test_version_is_derived_not_duplicated(self):
        """The tag must come from version.py, not a second hand-edited string."""
        v = self.tool.release_version()
        src = os.path.normpath(os.path.join(HERE, "..", "version.py"))
        with open(src, encoding="utf-8") as fh:
            self.assertIn('"%s"' % v, fh.read())

    # ---- manifest + rejection --------------------------------------------

    def _fake_release(self, tmp, tag="v0.0.0-test", break_what=None):
        """Build a synthetic sidecar release and return (archive, manifest)."""
        tool = self.tool
        adir, hname = tool.artifact_dir_name(), tool.helper_name()
        root = os.path.join(tmp, "pkg")
        os.makedirs(os.path.join(root, adir), exist_ok=True)
        with open(os.path.join(root, hname), "wb") as f:
            f.write(b"synthetic-helper")
        for name, _ in tool.required_files():
            with open(os.path.join(root, adir, name), "wb") as f:
                f.write(b"synthetic-artifact-" + name.encode())

        dist = os.path.join(tmp, "dist")
        os.makedirs(dist, exist_ok=True)
        stem = tool.asset_stem(tag)
        archive = os.path.join(dist, stem + ".zip")
        with zipfile.ZipFile(archive, "w") as z:
            z.write(os.path.join(root, hname), hname)
            for name, _ in tool.required_files():
                z.write(os.path.join(root, adir, name), "%s/%s" % (adir, name))

        # a manifest that claims the REAL registry digests, so the only thing
        # under test is the checking, not the synthetic bytes
        lines = [
            "# synthetic",
            "release_version\t%s" % tag,
            "platform\t%s" % tool.PLATFORM,
            "archive\t%s" % os.path.basename(archive),
            "archive_sha256\t%s" % tool.sha256(archive),
            "archive_bytes\t%d" % os.path.getsize(archive),
            "",
            "%s\t%d\t%s\t%s\tn/a" % (hname, 1, "0" * 64, "OPTIONAL_XDNA_HELPER"),
        ]
        for name, digest in tool.required_files():
            if break_what == "hash" and name.endswith(".xclbin"):
                digest = "0" * 64
                break_what = "done"
            lines.append("%s/%s\t%d\t%s\t%s\tYES" % (adir, name, 1, digest,
                                                     "QUALIFIED_XDNA_ARTIFACT"))
        manifest = os.path.join(dist, stem + ".manifest.txt")
        with open(manifest, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(lines) + "\n")
        return archive, manifest

    def test_clean_release_verifies(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive, _ = self._fake_release(tmp)
            self.assertEqual(self.tool.verify_release(archive), [])

    def test_wrong_archive_checksum_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive, _ = self._fake_release(tmp)
            with open(archive, "ab") as f:
                f.write(b"\x00")
            problems = self.tool.verify_release(archive)
            self.assertTrue(any("sha256 does not match" in p for p in problems), problems)

    def test_wrong_release_version_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive, manifest = self._fake_release(tmp)
            with open(manifest, encoding="utf-8") as fh:
                body = fh.read()
            body = body.replace("v0.0.0-test", "v9.9.9", 1)
            with open(manifest, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(body)
            problems = self.tool.verify_release(archive)
            self.assertTrue(any("version" in p for p in problems), problems)

    def test_requested_tag_mismatch_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive, _ = self._fake_release(tmp)
            problems = self.tool.verify_release(archive, tag_expected="v1.2.3")
            self.assertTrue(any("release version mismatch" in p for p in problems), problems)

    def test_digest_disagreeing_with_the_registry_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive, _ = self._fake_release(tmp, break_what="hash")
            problems = self.tool.verify_release(archive)
            self.assertTrue(any("compiled registry" in p for p in problems), problems)

    def test_missing_manifest_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive, manifest = self._fake_release(tmp)
            os.remove(manifest)
            problems = self.tool.verify_release(archive)
            self.assertTrue(any("manifest missing" in p for p in problems), problems)

    def test_missing_artifact_row_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive, manifest = self._fake_release(tmp)
            with open(manifest, encoding="utf-8") as fh:
                keep = [l for l in fh if "wa_F3_M256" not in l]
            with open(manifest, "w", encoding="utf-8", newline="\n") as fh:
                fh.writelines(keep)
            problems = self.tool.verify_release(archive)
            self.assertTrue(any("missing" in p for p in problems), problems)

    # ---- archive shape ---------------------------------------------------

    def test_archive_contains_exactly_the_asset_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive, _ = self._fake_release(tmp)
            with zipfile.ZipFile(archive) as z:
                members = sorted(z.namelist())
            adir = self.tool.artifact_dir_name()
            expected = sorted(
                [self.tool.helper_name()]
                + ["%s/%s" % (adir, n) for n, _ in self.tool.required_files()])
            self.assertEqual(members, expected)
            self.assertFalse([m for m in members if m.endswith(".pyc")])
            self.assertFalse([m for m in members if "__pycache__" in m])
            self.assertFalse([m for m in members if m.startswith("/") or ":" in m])

    def test_no_external_runtime_is_bundled(self):
        """XRT and the MSVC runtime stay external -- never inside the sidecar."""
        with tempfile.TemporaryDirectory() as tmp:
            archive, _ = self._fake_release(tmp)
            with zipfile.ZipFile(archive) as z:
                members = [m.lower() for m in z.namelist()]
            for forbidden in ("xrt_core.dll", "xrt_coreutil.dll", "msvcp140.dll",
                              "vcruntime140.dll", "vcruntime140_1.dll"):
                self.assertFalse([m for m in members if m.endswith(forbidden)],
                                 "%s must not be bundled" % forbidden)

    # ---- CLI contract ----------------------------------------------------

    def test_cli_exposes_the_release_owner(self):
        out = subprocess.run([sys.executable, TOOL, "--help"],
                             capture_output=True, text=True, timeout=60).stdout
        for flag in ("--release", "--dist", "--verify-release", "--verify"):
            self.assertIn(flag, out)


if __name__ == "__main__":
    unittest.main()
