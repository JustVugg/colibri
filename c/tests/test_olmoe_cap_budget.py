"""OLMoE sizes its expert cache from the RAM budget, and only when asked to.

Before #1443 the launcher forwarded `--ram` to three engines and olmoe was not
one of them, and "no explicit --cap" reached the engine as a constant eight
slots per layer. Eight knows nothing about the model or the machine: on a box
whose whole expert set fits in RAM it costs a factor of five, and on a small box
it is no safer than a budget-derived number would be.

What is asserted here is the contract, not a speed: the sentinel (cap 0) makes
the engine choose, an explicit cap is left alone, and the choice is bounded by
the budget it was given. The engine prints the derivation, so the assertions
read the line it prints rather than guessing at behaviour.

Needs a converted tiny OLMoE container and a built engine; both are what the
OLMoE tiny oracle job already builds.
"""
import os
import re
import subprocess
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
ENGINE = HERE / ("olmoe.exe" if os.name == "nt" else "olmoe")
FIXTURE = Path(os.environ.get("COLI_OLMOE_FIXTURE", ""))
CACHE_LINE = re.compile(r"^\[cache\] (\d+) slots/layer of (\d+) experts", re.M)


def derived_cap(ram_gb=None, cap="0"):
    """Run the engine far enough to print its cache line, then stop it."""
    environment = {**os.environ, "SNAP": str(FIXTURE), "SERVE": "1"}
    if ram_gb is not None:
        environment["RAM_GB"] = str(ram_gb)
    else:
        environment.pop("RAM_GB", None)
    finished = subprocess.run([str(ENGINE), cap, "8"], input=b"",
                              stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                              env=environment, timeout=120)
    match = CACHE_LINE.search(finished.stderr.decode("utf-8", "replace"))
    return (int(match.group(1)), int(match.group(2))) if match else (None, None)


@unittest.skipUnless(ENGINE.exists(), "olmoe engine is not built")
@unittest.skipUnless(FIXTURE.name and (FIXTURE / "config.json").is_file(),
                     "COLI_OLMOE_FIXTURE not set to a converted OLMoE container")
class OlmoeCapBudgetTest(unittest.TestCase):
    def test_an_explicit_cap_is_never_second_guessed(self):
        cap, _ = derived_cap(ram_gb=64, cap="4")
        self.assertIsNone(cap, "the engine re-derived a cap the caller had chosen")

    def test_a_generous_budget_holds_every_expert(self):
        cap, experts = derived_cap(ram_gb=64)
        self.assertIsNotNone(cap, "the sentinel produced no cache line")
        self.assertEqual(cap, experts,
                         "a budget with room for the whole expert set should hold it")

    def test_a_budget_below_the_floor_still_runs(self):
        """Not an error: one slot per layer is slow, refusing to start is worse."""
        cap, experts = derived_cap(ram_gb=0.4)
        self.assertEqual(cap, 1)
        self.assertGreater(experts, 1)

    def test_the_budget_bounds_the_choice(self):
        small, _ = derived_cap(ram_gb=0.4)
        large, _ = derived_cap(ram_gb=64)
        self.assertLess(small, large,
                        "the derived cap ignored the budget it was given")

    def test_without_a_budget_it_still_decides(self):
        cap, experts = derived_cap(ram_gb=None)
        self.assertIsNotNone(cap, "no RAM_GB must mean 'measure', not 'give up'")
        self.assertGreaterEqual(cap, 1)
        self.assertLessEqual(cap, experts)


if __name__ == "__main__":
    unittest.main()
