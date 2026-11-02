"""qwen36 on a container whose weights were chosen by someone else.

Two shapes the published advisories fixed in other engines and qwen36 never got:

  - an all-NaN router row. NaN > best is false for every expert, so the top-k pick
    stayed at -1 -- and route_select pads with -1 by design when fewer than K are
    eligible -- and expert_get() took -1 as an index and as a file offset
    (GHSA-5xpg-vw35-2687 in inkling, kimi_k3 and olmoe).
  - an e_score_correction_bias longer than n_experts. The buffer was n_experts
    floats; the copy was as long as the file said (GHSA-pmq2-6f2p-hjvf in inkling).

Needs the converted tiny container and its reference, which take PyTorch to build:
the Qwen3.6 tiny job of ci.yml makes them and runs this after its sanitizer build,
so a heap error is a diagnostic here and not a matter of luck.

  QWEN36_TINY_C=qwen36_tiny_c QWEN36_TINY_REF=qwen36_tiny/ref_full.json \
      python3 -m unittest -v tests.test_qwen36_hostile_weights
"""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from safetensors_edit import add_f32, copy_fixture, fill_nan  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
ENGINE = next((p for p in (HERE / "qwen36.exe", HERE / "qwen36") if p.exists()), None)
FIXTURE = Path(os.environ.get("QWEN36_TINY_C", ""))
REFERENCE = Path(os.environ.get("QWEN36_TINY_REF", ""))
SANITIZER = ("ERROR: AddressSanitizer", "runtime error:")


def run_engine(snapshot, **extra):
    environment = {**os.environ, "SNAP": str(snapshot), "COLI_DENSE_I8": "0", **extra}
    return subprocess.run([str(ENGINE), "1", "8", str(REFERENCE)], env=environment,
                          capture_output=True, text=True, errors="replace", timeout=600)


@unittest.skipUnless(ENGINE, "qwen36 engine is not built")
@unittest.skipUnless(FIXTURE.name and (FIXTURE / "config.json").is_file()
                     and REFERENCE.is_file(),
                     "QWEN36_TINY_C / QWEN36_TINY_REF not set to the converted tiny fixture")
class Qwen36HostileWeightsTest(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory()
        self.snapshot = copy_fixture(FIXTURE, Path(self.scratch.name) / "model")

    def tearDown(self):
        self.scratch.cleanup()

    def assertNoSanitizerReport(self, run):
        for marker in SANITIZER:
            self.assertNotIn(marker, run.stderr, run.stderr[-4000:])

    def test_an_all_nan_router_degrades_instead_of_indexing_minus_one(self):
        fill_nan(self.snapshot, "model.layers.0.mlp.gate.weight")
        run = run_engine(self.snapshot)
        self.assertNoSanitizerReport(run)
        # The tokens are wrong -- the router is garbage -- so the reference
        # comparison fails, but the process must finish rather than die on a signal.
        self.assertGreaterEqual(run.returncode, 0, run.stderr[-4000:])
        self.assertIn("[router] non-finite logits", run.stderr)

    def test_the_cache_route_pads_are_not_passed_on_as_minus_one(self):
        # route_select ranks nothing on a NaN row and pads all K slots with -1,
        # a different producer of the same id than the plain loop above.
        fill_nan(self.snapshot, "model.layers.0.mlp.gate.weight")
        run = run_engine(self.snapshot, CACHE_ROUTE="1")
        self.assertNoSanitizerReport(run)
        self.assertGreaterEqual(run.returncode, 0, run.stderr[-4000:])
        self.assertIn("[router] non-finite logits", run.stderr)

    def test_a_correction_bias_longer_than_n_experts_is_refused(self):
        add_f32(self.snapshot, "model.layers.0.mlp.gate.weight",
                "model.layers.0.mlp.gate.e_score_correction_bias", 4096)
        run = run_engine(self.snapshot)
        self.assertNoSanitizerReport(run)
        self.assertEqual(run.returncode, 1, run.stderr[-4000:])
        self.assertIn("e_score_correction_bias", run.stderr)
        self.assertIn("refusing", run.stderr)


if __name__ == "__main__":
    unittest.main()
