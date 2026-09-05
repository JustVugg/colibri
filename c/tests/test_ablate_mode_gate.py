"""The ablation scoring mode must stay unreachable unless it is asked for, and
its output must satisfy a checker written independently of the engine.

Two different risks are covered here, and neither is covered anywhere else.

The first is that the mode becomes reachable on a normal run.  Ablation is a
diagnostic path: it replaces the whole decode with a teacher-forced sweep and
returns its own process status.  If its dispatch ever ran unconditionally,
every ordinary invocation would stop doing what it was asked to do.  That
branch sits after the model is loaded, and this repository ships no weights, so
no test can reach it by running the engine; what can be checked, exactly and
mechanically, is that the engine contains no path into the mode other than its
environment variable.  These checks read `colibri.c` for that reason, and they
fail if the guard is removed, weakened, or bypassed by a second call site.

The second is that the engine and the offline checker drift apart.  The mode's
whole purpose is to produce an artifact that `tools/check_ablate_evidence.py`
can validate; a producer change that the checker would reject is a defect even
if the engine is self-consistent.  The last check runs the real producer and
the real checker against each other, once per manifest framing the engine
accepts, because the framings are exactly where the two could disagree: the
engine normalises them before hashing, and the checker has to reproduce that
normalisation rather than hash the file as it sits on disk.
"""

import pathlib
import subprocess
import sys
import tempfile
import unittest

HERE = pathlib.Path(__file__).resolve().parent
ENGINE = HERE.parent / "colibri.c"
VALIDATOR = HERE.parent / "tools" / "check_ablate_evidence.py"
PRODUCER = HERE / "test_ablate_mode"

ENV_GUARD = 'if(getenv("ABLATE_SCORE")){'
ADAPTER_DEFINE = "COLI_TEST_ABLATE_ADAPTERS"


def _engine_lines():
    return ENGINE.read_text(encoding="utf-8").splitlines()


def _adapter_only_lines(lines):
    """Line numbers (0-based) that the compiler only sees when the model-free
    adapter build is selected."""
    guarded = set()
    depth = 0
    adapter_depth = None
    for number, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("#if"):
            depth += 1
            if adapter_depth is None and ADAPTER_DEFINE in stripped:
                adapter_depth = depth
        elif stripped.startswith("#endif"):
            if adapter_depth is not None and depth == adapter_depth:
                adapter_depth = None
            depth = max(0, depth - 1)
        elif adapter_depth is not None:
            guarded.add(number)
    return guarded


class AblationModeEntryTest(unittest.TestCase):
    def test_the_mode_has_exactly_one_product_entry_and_it_is_the_guard(self):
        lines = _engine_lines()
        adapter_only = _adapter_only_lines(lines)
        uses = [
            number
            for number, line in enumerate(lines)
            if "ABLATE_MAIN_RETURN(" in line and not line.lstrip().startswith("#define")
        ]
        self.assertEqual(len(uses), 2, "expected one product entry and one adapter entry")
        product = [number for number in uses if number not in adapter_only]
        adapter = [number for number in uses if number in adapter_only]
        self.assertEqual(len(product), 1, "the ablation mode has more than one product entry")
        self.assertEqual(len(adapter), 1, "the adapter entry is no longer compile-gated")

        guard = lines[product[0] - 1].strip()
        self.assertEqual(
            guard,
            ENV_GUARD,
            "the product entry to the ablation mode is not guarded by its "
            "environment variable; an unguarded dispatch would replace every "
            "ordinary run with a diagnostic sweep",
        )

    def test_no_second_path_reaches_the_mode(self):
        lines = _engine_lines()
        dispatch = [line for line in lines if "ablate_mode_dispatch(" in line]
        self.assertEqual(
            [line for line in dispatch if line.lstrip().startswith("static int ablate_mode_dispatch")],
            [line for line in dispatch if "static int" in line],
            "ablate_mode_dispatch is declared more than once",
        )
        callers = [
            line
            for line in dispatch
            if "static int" not in line and not line.strip().startswith("*")
        ]
        self.assertEqual(
            len(callers), 1, "ablate_mode_dispatch is called from somewhere other than the guard macro"
        )
        self.assertIn("ablate_main_rc=ablate_mode_dispatch", callers[0].replace(" ", ""))

        runners = [
            number
            for number, line in enumerate(lines)
            if "run_ablate_score(" in line and "static int run_ablate_score" not in line
        ]
        self.assertEqual(len(runners), 1, "run_ablate_score is called from more than one place")
        self.assertIn(
            "ablate_model_mode_run",
            "\n".join(lines[runners[0] - 3 : runners[0] + 1]),
            "run_ablate_score is called from outside the mode implementation",
        )


class AblationEvidenceRoundTripTest(unittest.TestCase):
    FRAMINGS = ("lf", "crlf", "unterminated")

    def test_producer_output_passes_the_offline_checker(self):
        if not PRODUCER.exists():
            reason = (
                f"{PRODUCER.name} is not built, so real producer output was NOT "
                f"checked against the offline checker this run; build it with "
                f"`make {PRODUCER.relative_to(HERE.parent)}` and re-run"
            )
            # A silent skip here reads as a pass, and the thing being skipped is
            # the only check that the engine and the checker still agree. Say so
            # on the console as well as in the unittest result.
            print(f"SKIP: {reason}", file=sys.stderr, flush=True)
            self.skipTest(reason)
        digests = {}
        for framing in self.FRAMINGS:
            with self.subTest(framing=framing), \
                    tempfile.TemporaryDirectory() as directory:
                emitted = subprocess.run(
                    [str(PRODUCER), "--emit-round-trip", directory, framing],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(emitted.returncode, 0, emitted.stderr)
                checked = subprocess.run(
                    [
                        sys.executable,
                        str(VALIDATOR),
                        "--config",
                        f"{directory}/config.json",
                        f"{directory}/manifest.txt",
                        f"{directory}/evidence.jsonl",
                    ],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(
                    checked.returncode,
                    0,
                    f"the offline checker rejected real producer output "
                    f"({framing}):\n{checked.stdout}{checked.stderr}",
                )
                self.assertIn("PASS", checked.stdout)
                digests[framing] = checked.stdout.split("manifest=")[1].split()[0]
                # The framings must really differ on disk, or this proves nothing.
                raw = pathlib.Path(directory, "manifest.txt").read_bytes()
                if framing == "crlf":
                    self.assertIn(b"\r\n", raw)
                elif framing == "unterminated":
                    self.assertFalse(raw.endswith(b"\n"))
                else:
                    self.assertTrue(raw.endswith(b"\n"))
                    self.assertNotIn(b"\r", raw)
        self.assertEqual(
            len(set(digests.values())), 1,
            f"the framings bound different manifest digests: {digests}")


if __name__ == "__main__":
    unittest.main()
