"""qwen36 sizes its expert cache from the RAM budget when no explicit cap is
given, instead of the old hardcoded default of 16.

Before this change, bare omission of argv[1] (and the Segment API's
memory_limit_bytes==0 path) reached the engine as a constant that knew
nothing about the model or the machine. cap<=0 (explicit "0" or bare
omission) is now the sentinel that makes model_init_range derive a budget
from host RAM instead (qwen36_cap_for_ram(), see test_qwen36_cap_precedence.c
for the pure-function unit tests of the arithmetic itself).

What is asserted here is the contract, not a speed: the sentinel makes the
engine choose, an explicit cap is left alone, and the choice is bounded by
the budget it was given. The engine prints the derivation, so the assertions
read the line it prints rather than guessing at behaviour. Mirrors
test_olmoe_cap_budget.py's approach for the sibling engine.

Needs a converted tiny Qwen3.6 container and a built engine; set
COLI_QWEN36_FIXTURE to the container directory (same fixture convention as
test_qwen36_dashboard.py).
"""
import os
import re
import subprocess
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
ENGINE = HERE / ("qwen36.exe" if os.name == "nt" else "qwen36")
FIXTURE = Path(os.environ.get("COLI_QWEN36_FIXTURE", ""))
CACHE_LINE = re.compile(r"^\[qwen36\] cache auto-sized: (\d+) slots/layer of (\d+) experts", re.M)

# The tiny toy fixture (make_qwen36_tiny.py's default qwen36-35b geometry:
# hidden=64, inter=32, 8 experts) has a per-slot cost of only a few KB, so a
# RAM_GB comfortably below a real model's floor (e.g. 0.4, as the sibling
# OLMoE test uses) still clamps this fixture's derived cap at n_experts
# instead of flooring at 1. Use a value tiny enough to underflow even this
# toy geometry's floor.
BELOW_FLOOR_RAM_GB = 0.000001


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


@unittest.skipUnless(ENGINE.exists(), "qwen36 engine is not built")
@unittest.skipUnless(FIXTURE.name and (FIXTURE / "config.json").is_file(),
                     "COLI_QWEN36_FIXTURE not set to a converted Qwen3.6 container")
class Qwen36CapBudgetTest(unittest.TestCase):
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
        cap, experts = derived_cap(ram_gb=BELOW_FLOOR_RAM_GB)
        self.assertEqual(cap, 1)
        self.assertGreater(experts, 1)

    def test_the_budget_bounds_the_choice(self):
        small, _ = derived_cap(ram_gb=BELOW_FLOOR_RAM_GB)
        large, _ = derived_cap(ram_gb=64)
        self.assertLess(small, large,
                        "the derived cap ignored the budget it was given")

    def test_without_a_budget_it_still_decides(self):
        cap, experts = derived_cap(ram_gb=None)
        self.assertIsNotNone(cap, "no RAM_GB must mean 'measure', not 'give up'")
        self.assertGreaterEqual(cap, 1)
        self.assertLessEqual(cap, experts)

    def test_bare_omission_matches_explicit_zero(self):
        """The user's ask: the bare-CLI default (argv[1] omitted) must be the
        same sentinel as an explicit cap=0, not the old hardcoded 16. qwen36's
        argv is purely positional (no flags), so bits can only be given
        explicitly alongside an explicit cap -- the true bare invocation
        (argc==1) also leaves bits at its own default (4), so the equivalent
        explicit form is "0 4", not "0 8"."""
        environment = {**os.environ, "SNAP": str(FIXTURE), "SERVE": "1", "RAM_GB": "8"}
        bare = subprocess.run([str(ENGINE)], input=b"",
                              stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                              env=environment, timeout=120)
        explicit = subprocess.run([str(ENGINE), "0", "4"], input=b"",
                                  stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                  env=environment, timeout=120)
        bare_match = CACHE_LINE.search(bare.stderr.decode("utf-8", "replace"))
        explicit_match = CACHE_LINE.search(explicit.stderr.decode("utf-8", "replace"))
        self.assertIsNotNone(bare_match, "bare argv[1] omission produced no auto-sized line")
        self.assertIsNotNone(explicit_match, "explicit cap=0 produced no auto-sized line")
        self.assertEqual(bare_match.group(0), explicit_match.group(0))


if __name__ == "__main__":
    unittest.main()
