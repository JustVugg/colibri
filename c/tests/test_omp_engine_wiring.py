"""Every engine must size its OpenMP team, and the shared helper is how.

omp_tune.h has carried the physical-core sizing since #718 (+2.3x on a 16C/32T
part, from the thread count alone). Four engines adopted it; four more were
written afterwards and never did, including two of the newest. Nothing failed
when that happened, because the symptom is not an error: the engine simply runs
one thread per logical CPU, and on a host where the logical/physical ratio is
large the per-region barrier cost swallows the arithmetic. On a 208-logical-CPU
box deepseek-v41 spent 93% of its cycles inside libgomp and decoded 18.7x slower
than it does with a sane team (docs/experiments/dsv41-omp-team-2026-09-15.md).

Source-level, because the wiring is a line of code and a missing line is exactly
what has no runtime symptom. This generalizes tests/test_inkling_omp_source.py,
which guarded one engine against a defect that four had.
"""
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def strip_comments(source: str) -> str:
    """Drop /* */ and // comments before searching.

    Without this the gate matches a call that has been commented out -- which is
    how it was first written, and how the one engine it replaced would have
    passed a deliberately disabled call. Blank the comment out rather than
    deleting it so line-oriented patterns keep their shape.
    """
    source = re.sub(r"/\*.*?\*/", " ", source, flags=re.DOTALL)
    return re.sub(r"//[^\n]*", " ", source)


# engine source -> the label it passes to the helper
SHARED_HELPER = {
    "colibri.c": "colibri",
    "inkling.c": "inkling",
    "kimi_k3.c": "kimi_k3",
    "olmoe.c": "olmoe",
    "glm53.c": "glm53",
    "qwen36.c": "qwen36",
    "qwen38.c": "qwen38",
    "deepseek_v41.c": "deepseek-v41",
}

# deepseek_v4.c sizes its own team (loader workers reserved) rather than calling
# the shared helper, and that is a deliberate difference, not an omission; it
# still has to set a team size, or the same defect arrives through the side door.
OWN_SIZING = ("deepseek_v4.c",)


class OpenMPTeamSizingWiringTest(unittest.TestCase):
    def test_every_engine_includes_the_shared_helper(self):
        for name in SHARED_HELPER:
            with self.subTest(engine=name):
                source = strip_comments((ROOT / name).read_text(encoding="utf-8"))
                # assertTrue, not assertIn: assertIn prints the whole source on
                # failure, and these files are hundreds of kilobytes long.
                self.assertTrue('#include "omp_tune.h"' in source,
                                f"{name} does not include the shared OpenMP sizing helper")

    def test_every_engine_calls_it_with_its_own_label(self):
        for name, label in SHARED_HELPER.items():
            with self.subTest(engine=name):
                source = strip_comments((ROOT / name).read_text(encoding="utf-8"))
                self.assertTrue(f'coli_omp_tune_threads("{label}")' in source,
                                f"{name} never sizes its OpenMP team "
                                f"(expected coli_omp_tune_threads(\"{label}\"))")

    def test_the_call_is_reachable_from_main(self):
        """A call in a dead helper is the same defect with extra steps."""
        for name in SHARED_HELPER:
            with self.subTest(engine=name):
                source = (ROOT / name).read_text(encoding="utf-8")
                self.assertTrue(re.search(r'^int main\s*\(', source, re.MULTILINE),
                                f"{name} has no main()")

    def test_engines_with_their_own_sizing_still_set_a_team(self):
        for name in OWN_SIZING:
            with self.subTest(engine=name):
                source = strip_comments((ROOT / name).read_text(encoding="utf-8"))
                self.assertTrue("omp_set_num_threads" in source,
                                f"{name} neither calls the shared helper nor sets a team")


if __name__ == "__main__":
    unittest.main()
