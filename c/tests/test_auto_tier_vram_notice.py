"""--auto-tier must apply the VRAM tier it printed, or say why it cannot.

`coli plan` and `coli doctor` print a VRAM tier. --auto-tier is documented as
applying the plan those commands print. On the sibling-engine path it then
dropped the tier whenever COLI_CUDA was not already "1", which --auto-tier on
its own never sets, and it dropped it in silence: no [PLAN] line, no [CUDA]
banner, just half the throughput. On the box in #1581, 11.8 tok/s against 21.

The drop was deliberate once. cuda_binary() could only inspect the GLM binary,
so a working qwen36 CUDA build looked CPU-only to it and the safe move was to
require an explicit flag rather than risk launching a CPU-only sibling as a
CUDA one. Since #1533 the check takes the engine that will actually run, so
that risk is testable rather than assumed -- which is what these tests pin
down, in both directions:

  * a GPU build gets the tier without a flag, like the GLM path always has;
  * a CPU-only build still does not, and is told so;
  * --gpu none still wins over everything, and is told so differently, because
    it is the one of the two the user can undo by dropping a flag.

And the notice stays quiet when the tier was not real. A warning that fires on
every CPU-only launch is a warning people learn to scroll past.
"""
import contextlib
import importlib.machinery
import importlib.util
import io
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

_loader = importlib.machinery.SourceFileLoader("coli_cli_vram", str(HERE / "coli"))
_spec = importlib.util.spec_from_loader("coli_cli_vram", _loader)
coli = importlib.util.module_from_spec(_spec)
_loader.exec_module(coli)


class Args:
    def __init__(self, model="/models/qwen36"):
        self.model = model


def plan(devices, budget_bytes):
    return {"tiers": {"vram": {"devices": devices, "budget_bytes": budget_bytes}}}


def notice_for(p, gpu_off_by_request=False):
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        coli.report_unapplied_vram_tier(p, gpu_off_by_request)
    return err.getvalue()


QUALIFIED = [{"index": 0, "name": "NVIDIA GeForce RTX 4060 Ti",
              "free_bytes": 8 * 1000 ** 3}]


class PlanCudaEnabledTest(unittest.TestCase):
    """Which binary gets asked, and who wins when answers disagree."""

    def setUp(self):
        self.asked = []
        self._cuda_binary = coli.cuda_binary
        self._engine_for_gpu_check = coli.engine_for_gpu_check
        self._dsv4 = coli.dsv4_cuda_available
        coli.engine_for_gpu_check = lambda a: "/build/qwen36"

    def tearDown(self):
        coli.cuda_binary = self._cuda_binary
        coli.engine_for_gpu_check = self._engine_for_gpu_check
        coli.dsv4_cuda_available = self._dsv4

    def gpu_build(self, answer):
        def probe(engine=None):
            self.asked.append(engine)
            return answer
        coli.cuda_binary = probe

    def test_a_gpu_build_gets_the_tier_without_a_flag(self):
        """The defect in #1581: this used to be False, so the plan the user had
        just read was not the plan that ran."""
        self.gpu_build(True)
        self.assertTrue(coli.plan_cuda_enabled(Args(), "qwen36", {}))
        self.assertEqual(self.asked, ["/build/qwen36"],
                         "the family's own binary must be the one inspected; "
                         "asking colibri.exe about a qwen36 run is #1533")

    def test_a_cpu_only_build_still_does_not(self):
        self.gpu_build(False)
        self.assertFalse(coli.plan_cuda_enabled(Args(), "qwen36", {}))

    def test_gpu_none_wins_over_a_gpu_build(self):
        """COLI_CUDA=0 is what --gpu none sets, and it is the off switch."""
        self.gpu_build(True)
        self.assertFalse(coli.plan_cuda_enabled(Args(), "qwen36", {"COLI_CUDA": "0"}))
        self.assertEqual(self.asked, [], "an explicit off needs no probe")

    def test_an_explicit_gpu_flag_is_taken_at_its_word(self):
        """--gpu/--vram already validated the build and exited if it was not
        capable; probing again could only disagree with a decision made."""
        self.gpu_build(False)
        self.assertTrue(coli.plan_cuda_enabled(Args(), "qwen36", {"COLI_CUDA": "1"}))
        self.assertEqual(self.asked, [])

    def test_deepseek_v4_uses_its_own_probe(self):
        """V4 links the tier differently per platform; cuda_binary() would
        reject valid V4 CUDA builds (#1219)."""
        self.gpu_build(False)
        seen = []
        coli.dsv4_cuda_available = lambda model=None: (seen.append(model), True)[1]
        self.assertTrue(coli.plan_cuda_enabled(Args("/models/v4"), "deepseek_v4", {}))
        self.assertEqual(seen, ["/models/v4"])
        self.assertEqual(self.asked, [], "the generic probe must not be used for V4")


class VramNoticeTest(unittest.TestCase):
    def test_a_cpu_only_build_is_told_what_is_missing(self):
        out = notice_for(plan(QUALIFIED, 5_200_000_000))
        self.assertIn("VRAM tier", out)
        self.assertIn("5.2 GB", out, "the notice should say how much is unused")
        self.assertIn("NVIDIA GeForce RTX 4060 Ti", out, "and on which device")
        self.assertIn("CUDA=1", out, "and what would fix it")

    def test_gpu_none_is_told_something_it_can_act_on(self):
        """Dropping a flag and rebuilding an engine are not the same advice."""
        out = notice_for(plan(QUALIFIED, 5_200_000_000), gpu_off_by_request=True)
        self.assertIn("--gpu none", out)
        self.assertNotIn("CUDA=1", out,
                         "telling someone to rebuild when they asked for no GPU "
                         "sends them to fix something that is not broken")

    def test_an_unqualified_device_is_not_worth_a_warning(self):
        """free_bytes None means the planner never qualified that memory as a
        budget, so the tier was not going to be applied with the GPU on either."""
        out = notice_for(plan([{"index": 0, "name": "AMD", "free_bytes": None}],
                              5_200_000_000))
        self.assertEqual(out, "")

    def test_a_zero_budget_is_not_worth_a_warning(self):
        self.assertEqual(notice_for(plan(QUALIFIED, 0)), "")

    def test_no_devices_is_not_worth_a_warning(self):
        self.assertEqual(notice_for(plan([], 5_200_000_000)), "")

    def test_a_plan_of_the_wrong_shape_is_silent_and_harmless(self):
        """It only prints. It must never be why a launch stops."""
        for broken in ({}, {"tiers": {}}, {"tiers": {"vram": {}}},
                       {"tiers": {"vram": {"devices": ["not a dict"],
                                           "budget_bytes": 1}}},
                       {"tiers": {"vram": {"devices": None, "budget_bytes": 1}}}):
            with self.subTest(plan=broken):
                self.assertEqual(notice_for(broken), "")


if __name__ == "__main__":
    unittest.main()
