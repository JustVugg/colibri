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
import re
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
ADAPTER_STATE = "g_ablate_adapter_test"
ADAPTER_STATE_RE = re.compile(r"\b" + re.escape(ADAPTER_STATE) + r"\b")


def _engine_lines():
    return ENGINE.read_text(encoding="utf-8").splitlines()


def _strip_comments(lines):
    """Return *lines* with the text of `//` and `/* ... */` comments (including
    comments that span multiple lines) blanked out to spaces.

    Line count and column positions are preserved so callers can keep using
    0-based line numbers from the original file.  Without this, a reference
    to `g_ablate_adapter_test` inside a comment reads as a real use -- a
    single stray comment could satisfy the "population is nonzero" anchor
    below even after every real reference was deleted.
    """
    out_lines = []
    in_block = False
    for line in lines:
        out = []
        i, n = 0, len(line)
        while i < n:
            if in_block:
                end = line.find("*/", i)
                if end == -1:
                    out.append(" " * (n - i))
                    i = n
                else:
                    out.append(" " * (end + 2 - i))
                    i = end + 2
                    in_block = False
                continue
            line_comment = line.find("//", i)
            block_comment = line.find("/*", i)
            if line_comment == -1 and block_comment == -1:
                out.append(line[i:])
                i = n
            elif block_comment == -1 or (line_comment != -1 and line_comment < block_comment):
                out.append(line[i:line_comment])
                out.append(" " * (n - line_comment))
                i = n
            else:
                out.append(line[i:block_comment])
                i = block_comment + 2
                in_block = True
        out_lines.append("".join(out))
    return out_lines


def _adapter_state_uses(lines):
    """0-based line numbers with a code (non-comment) reference to the
    adapter-only test state."""
    code_lines = _strip_comments(lines)
    return [number for number, line in enumerate(code_lines) if ADAPTER_STATE_RE.search(line)]


def _adapter_only_lines(lines):
    """Line numbers (0-based) that the compiler only sees when the model-free
    adapter build is selected.

    Tracks `#if`/`#ifdef`/`#ifndef`/`#elif`/`#else`/`#endif` nesting depth.
    A block opened with `#ifdef COLI_TEST_ABLATE_ADAPTERS` (or an equivalent
    `#if` testing the define) puts its own arm -- up to the first `#elif` or
    `#else` at that same depth -- into the adapter-only region; that
    `#elif`/`#else` is the PRODUCT arm and ends the region immediately,
    it does not extend it.  A block opened with `#ifndef
    COLI_TEST_ABLATE_ADAPTERS` is the mirror image: its own arm runs whenever
    the adapter harness is *not* selected, i.e. it IS the product arm, so the
    adapter-only region only begins at that block's `#else` (there is no
    inverted equivalent of `#elif` handled here, since `#ifndef` takes no
    condition to test one against). Nesting inside an active region is
    unaffected: every line stays adapter-only until the region's own
    `#elif`/`#else`/`#endif` is reached.
    """
    guarded = set()
    depth = 0
    adapter_depth = None
    pending_ifndef_depth = None
    for number, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("#if"):
            depth += 1
            if ADAPTER_DEFINE in stripped:
                if stripped.startswith("#ifndef"):
                    if pending_ifndef_depth is None:
                        pending_ifndef_depth = depth
                elif adapter_depth is None:
                    adapter_depth = depth
        elif stripped.startswith("#elif") or stripped.startswith("#else"):
            if adapter_depth is not None and depth == adapter_depth:
                # The #ifdef ADAPTER_DEFINE arm ends here; #elif/#else is a
                # product arm regardless of what it itself tests.
                adapter_depth = None
            elif (
                pending_ifndef_depth is not None
                and depth == pending_ifndef_depth
                and stripped.startswith("#else")
            ):
                # The #ifndef ADAPTER_DEFINE arm (the product arm) ends here;
                # its #else is the adapter-only arm.
                adapter_depth = depth
                pending_ifndef_depth = None
        elif stripped.startswith("#endif"):
            if adapter_depth is not None and depth == adapter_depth:
                adapter_depth = None
            if pending_ifndef_depth is not None and depth == pending_ifndef_depth:
                pending_ifndef_depth = None
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
        self.assertEqual(len(uses), 1, "expected exactly one entry into the ablation mode")
        product = [number for number in uses if number not in adapter_only]
        self.assertEqual(len(product), 1, "the ablation mode has more than one product entry")

        guard = lines[product[0] - 1].strip()
        self.assertEqual(
            guard,
            ENV_GUARD,
            "the product entry to the ablation mode is not guarded by its "
            "environment variable; an unguarded dispatch would replace every "
            "ordinary run with a diagnostic sweep",
        )

    def test_every_adapter_state_reference_is_compile_gated(self):
        """The model-free adapter build is a test harness, not a product mode.

        Its state exists only so the parser/writer/status path can be driven
        with no weights; if any reference to it escaped the
        COLI_TEST_ABLATE_ADAPTERS gate, the shipped engine would carry a
        runtime test mode -- state that a normal run could observe or branch
        on.  Counting the gated references and requiring that count to be the
        whole population fails both ways it can break: a reference moved out of
        the gate, and the gate itself removed (which empties the gated set).
        """
        lines = _engine_lines()
        adapter_only = _adapter_only_lines(lines)
        uses = _adapter_state_uses(lines)
        self.assertGreater(
            len(uses), 0,
            f"{ADAPTER_STATE} is gone from the engine; this check no longer "
            f"guards anything and must be retired or re-pointed",
        )
        gated = [number for number in uses if number in adapter_only]
        self.assertEqual(
            len(gated), len(uses),
            f"{ADAPTER_STATE} is referenced outside the {ADAPTER_DEFINE} gate "
            f"at engine lines "
            f"{[number + 1 for number in uses if number not in adapter_only]}; "
            f"the product build must carry no adapter state",
        )

    def test_no_second_path_reaches_the_mode(self):
        lines = _engine_lines()
        entry = [line for line in lines if "ablate_model_mode_run(" in line]
        self.assertEqual(
            [line for line in entry if line.lstrip().startswith("static int ablate_model_mode_run")],
            [line for line in entry if "static int" in line],
            "ablate_model_mode_run is declared more than once",
        )
        callers = [
            line
            for line in entry
            if "static int" not in line and not line.strip().startswith("*")
        ]
        self.assertEqual(
            len(callers), 1,
            "ablate_model_mode_run is called from somewhere other than the guard macro",
        )
        self.assertIn("ablate_main_rc=ablate_model_mode_run", callers[0].replace(" ", ""))

        runners = [
            number
            for number, line in enumerate(lines)
            if "run_ablate_score(" in line and "static int run_ablate_score" not in line
        ]
        self.assertEqual(len(runners), 1, "run_ablate_score is called from more than one place")
        enclosing = ""
        for number in range(runners[0], -1, -1):
            stripped = lines[number].strip()
            if stripped.startswith("static ") and stripped.endswith("{"):
                enclosing = stripped
                break
        self.assertIn(
            "ablate_model_mode_run",
            enclosing,
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
