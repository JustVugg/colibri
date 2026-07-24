import importlib.util
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest import mock


C_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(C_DIR))
TEXTUAL_AVAILABLE = importlib.util.find_spec("textual") is not None

if TEXTUAL_AVAILABLE:
    from tools import capture_ramdisk_tui as capture
else:
    capture = None


@unittest.skipUnless(TEXTUAL_AVAILABLE, "Textual is not installed")
class RamdiskTuiCaptureFixtureTest(unittest.TestCase):
    def test_plan_is_one_blocker_free_shared_full_deployment(self):
        plan = capture.documentation_plan()

        self.assertEqual(plan["topology"], "interleaved")
        self.assertEqual(plan["mode"], "full")
        self.assertEqual(plan["blockers"], [])
        self.assertEqual(len(plan["mounts"]), 1)
        self.assertEqual(plan["mounts"][0]["path"], capture.SHARED_MOUNT)
        self.assertEqual(plan["placement"]["memory_nodes"], [0, 1])
        self.assertGreater(
            plan["reserve"]["available_bytes"],
            plan["reserve"]["total_required_bytes"],
        )
        self.assertEqual(
            plan["staging"]["staged_bytes"],
            plan["staging"]["total_staged_bytes"],
        )

    def test_lifecycle_double_refuses_host_reads_and_mutations(self):
        lifecycle = capture.RefusingLifecycle()

        for name in (
            *capture.RefusingLifecycle.HOST_READS,
            *capture.RefusingLifecycle.MUTATIONS,
        ):
            with self.subTest(name=name), self.assertRaises(
                capture.CaptureSafetyError
            ):
                getattr(lifecycle, name)()

        self.assertEqual(
            lifecycle.attempts,
            [
                *capture.RefusingLifecycle.HOST_READS,
                *capture.RefusingLifecycle.MUTATIONS,
            ],
        )

    def test_writer_emits_only_the_named_assets(self):
        screenshots = {
            name: '<svg width="1" height="1"></svg>'
            for name in capture.EXPECTED_ASSETS
        }
        bundle = capture.CaptureBundle(
            screenshots=screenshots,
            lifecycle_attempts=(),
            plan_token_reads=1,
            privilege_calls=0,
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "docs" / "media" / "ramdisk-tui"
            with mock.patch.object(capture, "ASSET_DIR", output):
                paths = capture.write_screenshots(bundle)

            self.assertEqual(
                tuple(path.name for path in paths),
                capture.EXPECTED_ASSETS,
            )
            self.assertEqual(
                {path.name for path in output.iterdir()},
                set(capture.EXPECTED_ASSETS),
            )


@unittest.skipUnless(TEXTUAL_AVAILABLE, "Textual is not installed")
class RamdiskTuiCaptureRenderTest(unittest.IsolatedAsyncioTestCase):
    async def test_capture_is_structural_deterministic_and_non_mutating(self):
        bundle = await capture.render_screenshots()

        self.assertEqual(
            tuple(bundle.screenshots),
            capture.EXPECTED_ASSETS,
        )
        self.assertEqual(bundle.lifecycle_attempts, ())
        self.assertEqual(bundle.plan_token_reads, 1)
        self.assertEqual(bundle.privilege_calls, 0)
        for name, svg in bundle.screenshots.items():
            with self.subTest(name=name):
                root = ET.fromstring(svg)
                self.assertTrue(root.tag.endswith("svg"))
                view_box = tuple(
                    float(value) for value in root.attrib["viewBox"].split()
                )
                self.assertGreater(view_box[2], 0)
                self.assertGreater(view_box[3], 0)
                self.assertNotIn("Resize to at least 72", svg)
                self.assertTrue(
                    all(line == line.rstrip() for line in svg.splitlines())
                )

        # Rich derives SVG identifiers from terminal content, so rerendering
        # the same state must be byte-for-byte stable.
        ready_again = await capture._capture_step(
            name="04-ready.svg",
            state="ready",
            step_key="6",
            expected_text=("State", "ready", "Deployment health", "verified"),
            title="Colibri RAM workspace — Ready",
            lifecycle=capture.RefusingLifecycle(),
            privilege=capture.NonAuthorizingPrivilege(),
            message=(
                "Preparation complete; staged weights and placement are verified."
            ),
        )
        self.assertEqual(bundle.screenshots["04-ready.svg"], ready_again)


if __name__ == "__main__":
    unittest.main()
