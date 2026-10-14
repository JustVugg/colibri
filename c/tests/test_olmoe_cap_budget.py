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
import json
import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
ENGINE = HERE / ("olmoe.exe" if os.name == "nt" else "olmoe")
FIXTURE = Path(os.environ.get("COLI_OLMOE_FIXTURE", ""))
CACHE_LINE = re.compile(r"^\[cache\] (\d+) slots/layer of (\d+) experts", re.M)
SHRINK_LINE = re.compile(r"^\[cache\] the KV reaches (\d+) positions: (\d+) slots/layer", re.M)


def derived_cap(ram_gb=None, cap="0", context=None):
    """Run the engine far enough to print its cache line, then stop it."""
    environment = {**os.environ, "SNAP": str(FIXTURE), "SERVE": "1"}
    if ram_gb is not None:
        environment["RAM_GB"] = str(ram_gb)
    else:
        environment.pop("RAM_GB", None)
    if context is not None:
        environment["CTX"] = str(context)
    else:
        environment.pop("CTX", None)
    finished = subprocess.run([str(ENGINE), cap, "8"], input=b"",
                              stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                              env=environment, timeout=120)
    match = CACHE_LINE.search(finished.stderr.decode("utf-8", "replace"))
    return (int(match.group(1)), int(match.group(2))) if match else (None, None)


def smallest_budget_for_every_expert():
    """The RAM_GB, to 10 kB, below which the cache stops holding every expert.

    The tiny model's whole expert set is ~213 kB and its KV at the default
    4096 positions 8.4 MB, so a budget a few hundred kB above this one is the
    regime where setting the KV aside would decide the cache."""
    low, high = 0.4, 0.6
    _, experts = derived_cap(ram_gb=high, context=1)
    assert derived_cap(ram_gb=high, context=1)[0] == experts
    assert derived_cap(ram_gb=low, context=1)[0] < experts
    while high - low > 1e-5:
        middle = (low + high) / 2
        if derived_cap(ram_gb=middle, context=1)[0] == experts:
            high = middle
        else:
            low = middle
    return high, experts


def generate(folder, tokens, ram_gb=None, cap="0"):
    """Greedy generation of `tokens` positions; returns every forward's logits
    (DUMP) and the engine's stderr."""
    reference = folder / "ref.json"
    prompt = [1, 2, 3]
    reference.write_text(json.dumps({"prompt_ids": prompt,
                                     "full_ids": prompt + [0] * (tokens - len(prompt))}))
    dump = folder / ("logits_%s_%s.bin" % (cap, ram_gb))
    environment = {**os.environ, "SNAP": str(FIXTURE), "DUMP": str(dump)}
    environment.pop("SERVE", None)
    environment.pop("CTX", None)
    if ram_gb is not None:
        environment["RAM_GB"] = str(ram_gb)
    else:
        environment.pop("RAM_GB", None)
    finished = subprocess.run([str(ENGINE), cap, "8", str(reference)],
                              stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                              env=environment, timeout=120)
    if finished.returncode != 0:
        raise AssertionError(finished.stderr.decode("utf-8", "replace"))
    return dump.read_bytes(), finished.stderr.decode("utf-8", "replace")


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

    def test_the_kv_is_not_set_aside_before_it_is_written(self):
        """The cache used to subtract the KV of the whole context (CTX) at load,
        before a single position was written; it now shares that room with the
        KV as the context grows."""
        budget, experts = smallest_budget_for_every_expert()
        budget += 0.002     # 2 MB: above the run-to-run noise, under the 8.4 MB KV
        self.assertEqual(derived_cap(ram_gb=budget, context=1)[0], experts)
        self.assertEqual(derived_cap(ram_gb=budget, context=4096)[0], experts,
                         "the cache set aside the KV of a context nothing has written")

    def test_the_kv_takes_its_slots_back_and_the_logits_do_not_move(self):
        """400 positions reach ~1 MB of KV pages, more than the 0.5 MB of room
        left over every expert: the cap must come down as the context grows,
        to the one-slot floor, with the logits of a run that held every expert
        all along."""
        budget, experts = smallest_budget_for_every_expert()
        with tempfile.TemporaryDirectory() as folder:
            shared, log = generate(Path(folder), 400, ram_gb=budget + 0.0005)
            alone, alone_log = generate(Path(folder), 400, cap=str(experts))
        caps = [int(found.group(2)) for found in SHRINK_LINE.finditer(log)]
        self.assertTrue(caps, "the KV never took a slot back:\n" + log)
        self.assertEqual(caps, sorted(caps, reverse=True), "the cap went back up")
        self.assertEqual(caps[-1], 1, "400 positions should leave the floor: " + str(caps))
        self.assertIsNone(SHRINK_LINE.search(alone_log), "an explicit cap was shrunk")
        self.assertGreater(len(alone), 0)
        self.assertEqual(shared, alone, "the logits moved with the cache's size")

    def test_without_a_budget_it_still_decides(self):
        cap, experts = derived_cap(ram_gb=None)
        self.assertIsNotNone(cap, "no RAM_GB must mean 'measure', not 'give up'")
        self.assertGreaterEqual(cap, 1)
        self.assertLessEqual(cap, experts)


if __name__ == "__main__":
    unittest.main()
