"""End-to-end RAM-disk zero-SSD-read contract test.

Launches the real Colibri engine on a tmpfs-backed int4 model and asserts the live
PROF output reports ~0 physical SSD reads and that COLI_RAMMAP actually bound
experts. This is the live-process proof the dependency-free unit suite fakes:
FakeEngine hardcodes physical_ssd_bytes=0 (test_ramdisk.py), so a green suite
says nothing about the real engine's wire output. This test does.

The acceptance contract being proven (docs/SETTINGS.md, "full mode"):
  * the whole model is staged on tmpfs, and
  * COLI_RAMMAP=1 binds every eligible expert as an immutable direct view, so
  * the decode window does zero block-device reads -> /proc/self/io read_bytes
    stays ~0, reported as `[PROF] physical SSD reads: 0.000 GB`.

Gating: needs distinct canonical and staged GLM-compatible int4 namespaces plus
the built colibri binary. The canonical namespace must be block-backed and have
its safetensor shards deliberately hidden after copying; the staged namespace
must be tmpfs-backed and complete. This proves the engine used
COLI_WEIGHTS_DIR rather than silently falling back to SNAP. The live test runs
only when both variables below are set; otherwise it skips honestly.

    COLI_RAMMAP_E2E_CANONICAL=/path/on/disk/glm_i4 \
    COLI_RAMMAP_E2E_STAGED=/dev/shm/glm_i4 \
      python3 -m pytest tests/test_rammap_e2e.py -v

The parse logic itself is covered by ProfParseTest below, which runs everywhere
and pins the exact colibri.c PROF emission strings.
"""

import os
import re
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENGINE = ROOT / "colibri"

# `[PROF] RAM map: <experts> experts / <gb> GB direct | ...` (colibri.c prof_report,
# printed only when COLI_RAMMAP bound at least one expert -> tmpfs-backed).
RAM_MAP_RE = re.compile(r"\[PROF\] RAM map: (\d+) experts / ([0-9.]+) GB direct")
# `[PROF] physical SSD reads: <gb> GB (...)` -- the live /proc/self/io read_bytes
# delta. The "unavailable" branch (non-Linux, or no /proc/self/io) has no number.
PHYSICAL_RE = re.compile(r"\[PROF\] physical SSD reads: ([0-9.]+) GB")
UNAVAILABLE = "physical SSD reads: unavailable"

# A fully-tmpfs model does zero block-device reads during the decode window, so
# read_bytes stays ~0. 50 MB is far above any kernel-accounting noise and far
# below the GBs real SSD streaming would report -- it discriminates the only
# regressions that matter here (model not on tmpfs, or RAMMAP silently bypassed
# in favor of a block-backed path). tmpfs slab fallback still reads ~0, which is
# correct: tmpfs is not a block device.
MAX_PHYSICAL_GB = 0.05
# REPLAY drives the forward from these token ids -- no tokenizer.json needed, so
# the test runs on the CI-generated bench fixture (random weights, no tokenizer)
# as well as on a real model. REF_FORCE bypasses the vocab-size sanity check.
REF = ROOT / "ref_glm.json"


def parse_prof(output):
    """Return (rammap_experts, rammap_gb, physical_gb_or_None) from glm PROF output.

    physical_gb is None when the engine reports accounting as unavailable.
    """
    rm = RAM_MAP_RE.search(output)
    pm = PHYSICAL_RE.search(output)
    experts = int(rm.group(1)) if rm else 0
    rammap_gb = float(rm.group(2)) if rm else 0.0
    physical = float(pm.group(1)) if pm else None
    return experts, rammap_gb, physical


@unittest.skipUnless(
    os.environ.get("COLI_RAMMAP_E2E_CANONICAL")
    and os.environ.get("COLI_RAMMAP_E2E_STAGED"),
    "set distinct COLI_RAMMAP_E2E_CANONICAL and COLI_RAMMAP_E2E_STAGED dirs",
)
class RammapE2ETest(unittest.TestCase):
    """Live Colibri on tmpfs: asserts the zero-physical-SSD-read contract holds."""

    def setUp(self):
        self.canonical = Path(
            os.environ["COLI_RAMMAP_E2E_CANONICAL"]
        ).resolve()
        self.staged = Path(os.environ["COLI_RAMMAP_E2E_STAGED"]).resolve()
        self.assertTrue(ENGINE.exists(), "colibri binary not built -- run `make colibri`")
        self.assertTrue(
            self.canonical.is_dir(),
            "canonical model dir missing: %s" % self.canonical,
        )
        self.assertTrue(
            self.staged.is_dir(),
            "staged model dir missing: %s" % self.staged,
        )
        self.assertTrue(REF.exists(), "missing %s -- run from a repo checkout" % REF)
        self.assertNotEqual(self.canonical, self.staged)
        self.assertNotEqual(
            self.canonical.stat().st_dev,
            self.staged.stat().st_dev,
            "canonical and staged fixtures share a backing device",
        )
        canonical_fs = self._filesystem_type(self.canonical)
        staged_fs = self._filesystem_type(self.staged)
        self.assertIn(
            canonical_fs,
            ("ext4", "xfs"),
            "canonical fixture is not block-backed: %s" % canonical_fs,
        )
        self.assertEqual(staged_fs, "tmpfs")
        self.assertFalse(
            any(self.canonical.glob("*.safetensors")),
            "canonical shards must be hidden after staging",
        )
        self.assertTrue(
            any(self.staged.glob("*.safetensors")),
            "staged tmpfs namespace has no safetensor shards",
        )

    def _filesystem_type(self, path):
        result = subprocess.run(
            ["findmnt", "-T", str(path), "-n", "-o", "FSTYPE"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertEqual(
            result.returncode,
            0,
            "cannot identify filesystem for %s: %s" % (path, result.stderr),
        )
        return result.stdout.strip()

    def _engine_environment(self, *, staged):
        env = dict(os.environ)
        env.update(
            SNAP=str(self.canonical),
            COLI_RAMMAP="1",
            PROF="1",
            REPLAY="1",
            REF_FORCE="1",
            REF=str(REF),
            COLI_NO_OMP_TUNE="1",
        )
        if staged:
            env["COLI_WEIGHTS_DIR"] = str(self.staged)
        else:
            env.pop("COLI_WEIGHTS_DIR", None)
            env.pop("COLI_MODEL_DIRS", None)
            env.pop("COLI_MODEL_MIRROR", None)
        env.pop("COLI_STATE_DIR", None)
        return env

    def test_canonical_namespace_cannot_launch_without_staged_redirect(self):
        proc = subprocess.run(
            [str(ENGINE)],
            env=self._engine_environment(staged=False),
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertNotEqual(
            proc.returncode,
            0,
            "engine unexpectedly found poisoned canonical shards; the positive "
            "test would not prove COLI_WEIGHTS_DIR redirection",
        )

    def test_zero_physical_ssd_reads_on_tmpfs_rammap(self):
        # REPLAY mode drives a real forward (MoE expert loading -> RAMMAP -> I/O)
        # from the repo's ref_glm.json token ids. It needs no tokenizer.json, so it
        # works on the CI-generated bench fixture as well as on a real model. The
        # physical-read accounting is identical to the PROMPT/serve path: the same
        # prof_report emits the [PROF] lines this test parses.
        proc = subprocess.run(
            [str(ENGINE)],
            env=self._engine_environment(staged=True),
            capture_output=True,
            text=True,
            timeout=180,
        )
        output = proc.stdout + proc.stderr
        self.assertEqual(
            proc.returncode, 0, "colibri exited %d:\n%s" % (proc.returncode, output[-2000:])
        )
        self.assertNotIn(
            UNAVAILABLE,
            output,
            "physical SSD accounting unavailable -- not a Linux tmpfs backing?",
        )
        experts, _, physical = parse_prof(output)
        self.assertGreater(
            experts,
            0,
            "COLI_RAMMAP bound no experts -- model is not tmpfs-backed, so this "
            "run exercised the SSD path and proves nothing",
        )
        # The None branch is the "unavailable" case (already ruled out above); the
        # `if ... is None: fail()` narrows physical to float for the assertLess.
        if physical is None:
            self.fail("missing [PROF] physical SSD reads line")
        self.assertLess(
            physical,
            MAX_PHYSICAL_GB,
            "tmpfs + RAMMAP model still read %.3f GB from SSD during decode -- "
            "zero-SSD-read contract broken" % physical,
        )


class ProfParseTest(unittest.TestCase):
    """Pins parse_prof against the exact colibri.c PROF strings -- runs everywhere."""

    def test_parses_bound_tmpfs_output(self):
        sample = (
            "[PROF] RAM map: 4096 experts / 12.345 GB direct | 99 calls this window | zero slab reads\n"
            "[PROF] physical SSD reads: 0.000 GB (0.0 MB/token; Linux /proc/self/io read_bytes, process-wide)\n"
        )
        experts, rammap_gb, physical = parse_prof(sample)
        self.assertEqual(experts, 4096)
        self.assertAlmostEqual(rammap_gb, 12.345)
        self.assertEqual(physical, 0.000)

    def test_parses_nonzero_ssd_output(self):
        sample = "[PROF] physical SSD reads: 5.500 GB (0.2 MB/token; ...)\n"
        _, _, physical = parse_prof(sample)
        self.assertEqual(physical, 5.500)

    def test_parses_unavailable_as_none(self):
        sample = "[PROF] physical SSD reads: unavailable on this platform/kernel\n"
        _, _, physical = parse_prof(sample)
        self.assertIsNone(physical)

    def test_detects_ssd_regression_threshold(self):
        # A real SSD-streaming regression reports GBs, well above the threshold.
        sample = "[PROF] physical SSD reads: 5.500 GB (...)\n"
        _, _, physical = parse_prof(sample)
        if physical is None:
            self.fail("physical SSD reads line did not parse")
        self.assertGreaterEqual(physical, MAX_PHYSICAL_GB)


if __name__ == "__main__":
    unittest.main()
