"""The Linux release archive must start on Ubuntu 24.04 (#2556).

A binary may need any GLIBC symbol version up to the glibc it was linked
against. release.yml built linux-x86_64 on `ubuntu-latest`, and once that
label meant Ubuntu 26.04 (glibc 2.43) the v2.1.0 archive stopped with
"GLIBC_2.43 not found" on Ubuntu 24.04 (glibc 2.39). Nothing in the workflow
looked at those versions: the verify steps ran the TUI and the engine on the
runner that had just built them, where they always start.

This pins the fix on the real workflow:

- every Linux row runs on a fixed image, not a moving `-latest` label, and
  declares the glibc the archive is held to, no newer than 24.04's;
- every `matrix.os` condition names an `os` the matrix has, so the apt steps
  cannot be skipped by a runner renamed in one place only;
- a verify step, gated on that `glibc`, reads the GLIBC versions every ELF
  file of the unpacked archive needs and fails on a newer one. Its shell is
  run under bash on small archives, with a stand-in for readelf.
"""
import os
import pathlib
import re
import shutil
import subprocess
import tempfile
import textwrap
import unittest

RELEASE = (pathlib.Path(__file__).resolve().parents[2] / ".github" / "workflows"
           / "release.yml")

#: Ubuntu 24.04's glibc: the oldest the Linux archive is promised to start on.
GLIBC_FLOOR = (2, 39)


def build_job(text):
    """The `build:` job of release.yml, up to the next job."""
    m = re.search(r"(?m)^  build:\n(.*?)(?=^  \S|\Z)", text, re.S)
    return m.group(1) if m else ""


def matrix_rows(job):
    """The build matrix's `include:` rows, each as {key: value}."""
    m = re.search(r"(?m)^        include:\n(.*?)(?=^    \S)", job, re.S)
    rows = []
    for chunk in re.split(r"(?m)^          - ", m.group(1) if m else "")[1:]:
        row = {}
        for line in chunk.splitlines():
            kv = re.match(r"\s*([a-z_0-9]+):\s*(.*)$", line)
            if kv:
                row[kv.group(1)] = kv.group(2).strip().strip('"')
        rows.append(row)
    return rows


def steps(job):
    """The build job's steps, as their source text, in order."""
    return re.split(r"(?m)^      - ", job)[1:]


class LinuxArchiveGlibcFloor(unittest.TestCase):
    def setUp(self):
        self.job = build_job(RELEASE.read_text(encoding="utf-8"))
        self.assertTrue(self.job, "release.yml: the build job is gone")
        self.rows = matrix_rows(self.job)
        self.linux = [r for r in self.rows if r.get("name", "").startswith("linux-")]
        self.assertIn("linux-x86_64", [r["name"] for r in self.linux],
                      "release.yml: no linux-x86_64 row")

    def test_every_linux_row_runs_on_a_fixed_image_with_its_floor(self):
        for row in self.linux:
            name, os_label = row["name"], row.get("os", "")
            with self.subTest(name=name):
                self.assertRegex(os_label, r"^ubuntu-\d\d\.\d\d(-arm)?$",
                                 f"{name} runs on '{os_label}': a moving label "
                                 f"changes the glibc the archive needs under a tag build")
                self.assertIn("glibc", row,
                              f"{name} declares no glibc: nothing holds the archive to one")
                floor = tuple(int(p) for p in row["glibc"].split("."))
                self.assertLessEqual(floor, GLIBC_FLOOR,
                                     f"glibc {row['glibc']} is newer than Ubuntu "
                                     f"24.04's: the archive would not start there")

    def test_every_os_condition_names_a_matrix_row(self):
        named = {r.get("os") for r in self.rows}
        conditions = re.findall(r"matrix\.os [!=]= '([^']+)'", self.job)
        self.assertTrue(conditions, "no matrix.os condition found: the test reads nothing")
        for label in conditions:
            self.assertIn(label, named,
                          f"a step compares matrix.os to '{label}', which no row "
                          f"has: that step never runs")

    def test_a_verify_step_reads_every_elf_file_against_the_floor(self):
        all_steps = steps(self.job)
        gated = [i for i, s in enumerate(all_steps) if "if: matrix.glibc != ''" in s]
        self.assertEqual(len(gated), 1, "no single step gated on matrix.glibc")
        step = all_steps[gated[0]]
        for needle, why in (
                ("cd verify", "it must read the unpacked archive, not the build tree"),
                ("find . -type f", "it must read every file, not a hand-kept list"),
                ("readelf -V --wide", "it must read the version references the loader checks")):
            self.assertIn(needle, step, f"the glibc step: {why}")
        unpacked = [i for i, s in enumerate(all_steps)
                    if s.startswith("name: Verify the packaged archive actually runs")]
        upload = [i for i, s in enumerate(all_steps)
                  if s.startswith("uses: actions/upload-artifact@")]
        self.assertTrue(unpacked and upload, "release.yml: verify or upload step not found")
        self.assertLess(unpacked[0], gated[0], "the glibc step runs before the archive is unpacked")
        self.assertLess(gated[0], upload[0], "the glibc step runs after the upload")


# Windows: shutil.which("bash") can return System32's WSL shim, which has no
# sed/tail. Prefer the bash of the Git for Windows that provides `git`
# (<root>\cmd\git.exe next to <root>\bin\bash.exe); never take a System32 one.
def _find_bash():
    git = shutil.which("git")
    candidates = [str(pathlib.Path(git).parents[1] / "bin" / "bash.exe") if git else None,
                  shutil.which("bash")]
    return next((c for c in candidates
                 if c and os.path.exists(c) and "system32" not in c.lower()), None)


BASH = _find_bash()

# A stand-in for binutils' readelf, enough for the step: a file whose first line is
# FAKEELF is an ELF file, and the rest of it is what `readelf -V --wide` prints for it;
# FAKEFAIL is an ELF file that `readelf -V` cannot read (exit 1). An empty static library
# (`!<arch>` alone) exits 0 and prints nothing, as binutils' readelf does.
FAKE_READELF = """#!/bin/sh
for f; do :; done
first=$(head -n 1 "$f" 2>/dev/null) || exit 1
case $first in FAKEELF|FAKEFAIL) ;; '!<arch>') exit 0 ;; *) exit 1 ;; esac
[ "$1" = -h ] && exit 0
[ "$first" = FAKEFAIL ] && exit 1
tail -n +2 "$f"
"""


def readelf_v(needs=(), defines=()):
    """`readelf -V --wide` as binutils 2.42 prints it (process_version_sections): the
    versym section, then the version definitions and the version needs (libc.so.6)."""
    if not needs and not defines:
        return "\nNo version information found in this file.\n"
    names = list(defines) + list(needs)
    out = (f"\nVersion symbols section '.gnu.version' contains {len(names) + 1} entries:\n"
           " Addr: 0x0000000000000610  Offset: 0x000610  Link: 6 (.dynsym)\n"
           "  000:   0 (*local*)    " +
           "".join(f"  {i + 2} ({name})" for i, name in enumerate(names)) + "\n")
    if defines:
        out += (f"\nVersion definition section '.gnu.version_d' contains {len(defines) + 1} entries:\n"
                " Addr: 0x0000000000000640  Offset: 0x000640  Link: 7 (.dynstr)\n"
                "  000000: Rev: 1  Flags: BASE  Index: 1  Cnt: 1  Name: libfixture.so\n" +
                "".join(f"  {0x1c * (i + 1):#06x}: Rev: 1  Flags: none  Index: {i + 2}  Cnt: 1  "
                        f"Name: {name}\n" for i, name in enumerate(defines)))
    if needs:
        out += (f"\nVersion needs section '.gnu.version_r' contains 1 entry:\n"
                " Addr: 0x0000000000000680  Offset: 0x000680  Link: 7 (.dynstr)\n"
                f"  000000: Version: 1  File: libc.so.6  Cnt: {len(needs)}\n" +
                "".join(f"  {0x10 * (i + 1):#06x}:   Name: {name}  Flags: none  "
                        f"Version: {i + 2 + len(defines)}\n" for i, name in enumerate(needs)))
    return out


@unittest.skipUnless(BASH, "the glibc step is a bash script")
class GlibcStepRuns(unittest.TestCase):
    """Run the step's own shell, as GitHub runs `shell: bash` (bash --noprofile --norc
    -eo pipefail), on small archives, with the readelf stand-in above on PATH."""

    @classmethod
    def setUpClass(cls):
        text = RELEASE.read_text(encoding="utf-8")
        job = build_job(text)
        [step] = [s for s in steps(job) if "if: matrix.glibc != ''" in s]
        body = step.split("run: |\n", 1)[1]
        cls.floor = next(r["glibc"] for r in matrix_rows(job) if r.get("name") == "linux-x86_64")
        cls.script = textwrap.dedent(body).replace("${{ matrix.glibc }}", cls.floor)

    def run_step(self, files):
        """files: {path inside the archive: content}. Returns (exit status, output)."""
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            (root / "bin").mkdir()
            fake = root / "bin" / "readelf"
            fake.write_text(FAKE_READELF, newline="\n")
            fake.chmod(0o755)
            for name, content in files.items():
                path = root / "verify" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, newline="\n")
            (root / "verify").mkdir(exist_ok=True)
            result = subprocess.run(
                [BASH, "--noprofile", "--norc", "-eo", "pipefail", "-c",
                 'PATH="$PWD/bin:$PATH"\n' + self.script],
                cwd=root, capture_output=True, text=True, check=False)
            return result.returncode, result.stdout + result.stderr

    def elf(self, needs=(), defines=()):
        return "FAKEELF\n" + readelf_v(needs, defines)

    def archive(self, **extra):
        files = {"colibri": self.elf(("GLIBC_2.2.5", "GLIBC_2.34")),
                 "qwen36": self.elf(("GLIBC_2.3", self.floor_version())),
                 "dir with space/olmoe": self.elf(("GLIBC_2.17",)),
                 "backend.o": self.elf(), "coli": "#!/usr/bin/env python3\n",
                 "shaders/x.spv": "\x03\x02\x23\x07"}
        files.update(extra)
        return files

    def floor_version(self):
        return "GLIBC_" + self.floor

    def test_an_archive_within_the_floor_passes(self):
        code, out = self.run_step(self.archive())
        self.assertEqual(code, 0, out)
        self.assertIn(f"OK: 3 of 4 ELF files link glibc, the newest version they need is "
                      f"{self.floor} (at most {self.floor})", out)

    def test_a_newer_version_fails_and_names_only_its_file(self):
        code, out = self.run_step(self.archive(**{"tui": self.elf(("GLIBC_2.9", "GLIBC_2.43"))}))
        self.assertEqual(code, 1, out)
        self.assertIn(f"FAIL: newer than glibc {self.floor}: ./tui (GLIBC_2.43)", out)
        self.assertNotIn("./colibri", out)

    def test_versions_compare_as_numbers(self):
        # 2.10 is newer than 2.9, and 2.34 than 2.7: as text, 2.10 and 2.34 sort first
        # (within one file, across files, and against the floor)
        for files, newest in (({"a": self.elf(("GLIBC_2.9", "GLIBC_2.10", "GLIBC_2.2.5"))}, "2.10"),
                              ({"libold.so": self.elf(("GLIBC_2.2.5", "GLIBC_2.7")),
                                "colibri": self.elf(("GLIBC_2.34",))}, "2.34")):
            with self.subTest(files=list(files)):
                code, out = self.run_step(files)
                self.assertEqual(code, 0, out)
                self.assertIn(f"the newest version they need is {newest}", out)

    def test_a_static_library_is_skipped(self):
        # readelf exits 0 and prints nothing for an empty one: not a damaged file
        code, out = self.run_step(self.archive(**{"lib/libcoli.a": "!<arch>\n"}))
        self.assertEqual(code, 0, out)
        self.assertIn("OK: 3 of 4 ELF files", out)

    def test_a_version_the_file_defines_is_not_a_need(self):
        code, out = self.run_step({"lib.so": self.elf(("GLIBC_2.2.5",), defines=("GLIBC_2.99",))})
        self.assertEqual(code, 0, out)
        self.assertIn("the newest version they need is 2.2.5", out)

    def test_no_glibc_version_at_all_fails(self):
        for files in ({"static": self.elf(), "backend.o": self.elf()}, {"coli": "text\n"}):
            with self.subTest(files=list(files)):
                code, out = self.run_step(files)
                self.assertEqual(code, 1, out)
                self.assertIn("FAIL: no GLIBC version in the archive's", out)

    def test_a_file_readelf_cannot_read_fails(self):
        # readelf exits 1; or, on a damaged file, exits 0 and prints no version section
        for content in ("FAKEFAIL\n", "FAKEELF\n"):
            with self.subTest(content=content):
                code, out = self.run_step(self.archive(broken=content))
                self.assertEqual(code, 1, out)
                self.assertIn("FAIL: readelf cannot read the versions of ./broken", out)


if __name__ == "__main__":
    unittest.main()
