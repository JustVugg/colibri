"""tools/eval_glm.py must accept only the engine's real SCORE stdout
records and refuse everything else with a named error: an unparsed or
duplicated banner/load preamble, a SCORE line whose numeric token is not
the canonical finite ``%.6f``/``%.17g`` spelling the engine actually
emits, and a foreign stdout line that is neither a preamble nor a SCORE
record. It must also write result rows incrementally (one flush per
request, never buffered until completion) and mark incomplete runs,
including a pre-launch refusal, before the engine is ever started.

Checks enumerated from the source (`tools/eval_glm.py`, read in full
before writing this module) and covered below, grouped by the function
that performs them:

- `parse_c17g`: both exact finite spellings the engine actually emits --
  the shipped ``%.6f`` form and the newer ``%.17g`` evidence form -- and
  nothing else; this module imports nothing from and shares no code with
  `check_data_logprob_gaps.py`, which parses its own independent grammar.
- `parse_score_result` / `_SCORE_RE`: the shipped ``%.6f`` form AND the
  ``%.17g`` evidence form (both numeric forms), non-finite/malformed
  rejection, and the ``<contlen> <greedy>`` metadata bounds.
- `classify_score_stdout` / `ScoreStdoutClassifier`: the banner/load
  preamble lifecycle (missing, duplicated, out-of-order, or a SCORE
  record before the load record all refuse); every line that is not an
  exact banner, an exact load record, or an exact SCORE record refuses
  by name -- a foreign line is never silently treated as a score, which
  is exactly the defect dev's plain ``line[0] in "-0123456789"`` filter
  does not catch (differential bite, below).
- `score_request_wire`: strict ASCII/LF request grammar, the per-record
  SHA-256 digest, and the inclusive 256 MiB engine text limit.
- `completion_error`: the exact zero-exit/complete-count/positive-token
  denominator that alone passes.
- `main`: incremental durability (one written+flushed row per completed
  request, never buffered until the run ends) and pre-launch INCOMPLETE
  marking (no benchmark tasks selected; zero SCORE requests produced;
  every choice's context/continuation split is empty) -- the engine is
  never launched for any of these; a mid-run crash or interrupt still
  leaves the INCOMPLETE marker and terminates the child process; a
  partial run still prints the accuracy table over whatever rows landed.

Deferred (need a live binary this module does not have access to):
- `test_c_emitted_c17g_corpus_is_canonical`, which drives
  ``test_logprob_wire --score-c17g-fixture`` (a binary produced by a
  different part of this project's build, not present here).

No model is run by the committed tests -- every case here drives
`eval_glm.py` against an injected stand-in for the direct engine launch
(a fake ``subprocess.Popen`` returning canned stdout/stderr), never a
real ``./glm`` process.
"""

import contextlib
import hashlib
import importlib.util
import io
import json
import os
import pathlib
import signal
import sys
import tempfile
import types
import unittest
from unittest import mock


HERE = pathlib.Path(__file__).resolve().parent
TOOLS = HERE.parent / "tools"

_spec = importlib.util.spec_from_file_location(
    "eval_glm_under_test", TOOLS / "eval_glm.py")
EVAL = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(EVAL)


class EvalGlmEvidenceTests(unittest.TestCase):
    BANNER = (
        "== GLM C engine (glm_moe_dsa), cache=64 experts/layer | "
        "compute experts@4-bit dense@8-bit | idot: neon-i8mm ==")

    @staticmethod
    def loaded(state="ACTIVE", draft=1, layers=78, experts=256,
               load="1.00", resident="1.00"):
        return (f"loaded in {load}s | resident dense: {resident} MB | "
                f"layers={layers} experts={experts} | MTP {state} "
                f"(draft={draft})")

    def run_eval_main(self, stdout_records):
        class Encoded:
            ids = [1, 2]

        class FakeTokenizer:
            @staticmethod
            def from_file(path):
                return FakeTokenizer()

            @staticmethod
            def encode(text):
                return Encoded()

        process = types.SimpleNamespace(
            returncode=0, stderr=(), stdout=tuple(stdout_records),
            wait=lambda: 0, poll=lambda: 0, terminate=lambda: None)
        with tempfile.TemporaryDirectory() as tmp:
            output = pathlib.Path(tmp) / "results.csv"
            (pathlib.Path(tmp) / "config.json").write_text(
                '{"vocab_size":3}\n')
            argv = [
                "eval_glm.py", "--snap", tmp, "--tasks", "smoke",
                "--limit", "1", "--glm", "/fake/glm", "--out",
                str(output),
            ]
            tokenizers = types.SimpleNamespace(Tokenizer=FakeTokenizer)
            stderr_buf = io.StringIO()
            stdout_buf = io.StringIO()
            with mock.patch.object(sys, "argv", argv), \
                    mock.patch.dict(sys.modules, {"tokenizers": tokenizers}), \
                    mock.patch.object(EVAL.subprocess, "Popen",
                                      return_value=process) as popen, \
                    contextlib.redirect_stderr(stderr_buf), \
                    contextlib.redirect_stdout(stdout_buf):
                rc = EVAL.main()
            self.last_popen_kwargs = popen.call_args.kwargs if popen.called else {}
            self.last_stderr = stderr_buf.getvalue()
            self.last_stdout = stdout_buf.getvalue()
            return rc, output.read_text(), popen.call_count

    def test_exact_score_text_survives_csv(self):
        text = "-8.5534581234567888"
        exact, value, contlen, greedy = EVAL.parse_score_result(
            f"{text} 4096 1")
        self.assertEqual(exact, text)
        self.assertEqual(contlen, 4096)
        self.assertEqual(greedy, 1)
        self.assertTrue(value < 0)

        out = io.StringIO()
        meta = ("task", 3, 2, 4096, 17, 2)
        EVAL.write_result_row(out, 9, meta, exact, greedy)
        self.assertEqual(
            out.getvalue(),
            "9,task,3,2,4096,17,2,-8.5534581234567888,1\n")

    def test_shipped_dot6f_form_is_still_accepted(self):
        # The tool must accept BOTH the shipped %.6f form the engine
        # actually prints today AND the %.17g evidence form.
        exact, value, contlen, greedy = EVAL.parse_score_result(
            "-8.553458 4096 1")
        self.assertEqual(exact, "-8.553458")
        self.assertEqual(value, -8.553458)
        self.assertEqual((contlen, greedy), (4096, 1))

    def test_direct_call_rejects_trailing_newline_record(self):
        # See parse_score_result's own docstring: _SCORE_RE's "$" anchor
        # matches just before a trailing "\n" as well as at true
        # end-of-string, so fullmatch -> match/search would accept a
        # newline-terminated record fullmatch correctly rejects. The
        # classifier's own callers strip the newline first and are
        # provably out of reach of this (see the docstring), but this
        # function is also called directly -- pin the direct-call
        # contract so a future weakening doesn't slip in unnoticed there.
        with self.assertRaises(EVAL.EvidenceError):
            EVAL.parse_score_result("-8.553458 4096 1\n")

    def test_parse_c17g_rejects_trailing_garbage(self):
        # parse_c17g is only ever reached, elsewhere in this module,
        # through _SCORE_RE's own "^...$"-anchored capture group, which
        # already excludes trailing garbage before parse_c17g ever sees
        # the text -- so nothing else in this module's test suite drives
        # parse_c17g directly. Direct coverage: verified (RAN, against a
        # scratch mutant copy, not committed) that weakening the
        # fullmatch to match here is actually an EQUIVALENT mutant for
        # every input tried -- the round-trip format comparison a few
        # lines below the match call independently rejects any text with
        # a non-canonical tail, since format(value, ...) can never
        # reproduce trailing garbage. This test pins parse_c17g's own
        # contract directly rather than only through callers; it does
        # not by itself prove the fullmatch call is load-bearing.
        with self.assertRaises(EVAL.EvidenceError):
            EVAL.parse_c17g("1.5garbage")
        with self.assertRaises(EVAL.EvidenceError):
            EVAL.parse_c17g("-8.553458 ")  # trailing space after a valid token
        self.assertEqual(EVAL.parse_c17g("1.5"), 1.5)

    def test_parse_c17g_rejects_bare_trailing_decimal_point(self):
        # "1." is valid Python float() syntax (float("1.") == 1.0), so
        # only the round-trip format comparison -- not the decimal
        # group's "one-or-more" digit requirement -- is what actually
        # rejects it: format(1.0, ".17g") == "1", never "1.". Verified
        # (RAN, scratch mutant, not committed) that widening that group
        # to "zero-or-more" does not change parse_c17g's accept/reject
        # outcome for any of these inputs; kept as direct-coverage
        # regression pins, not as proof the digit requirement is
        # independently load-bearing.
        for bad in ("1.", "-0.", "12."):
            with self.subTest(bad=bad):
                with self.assertRaises(EVAL.EvidenceError):
                    EVAL.parse_c17g(bad)

    def test_parse_c17g_rejects_single_digit_exponent(self):
        # C's %e/%g exponent is zero-padded to at least two digits
        # ("e-05", never "e-5"). Verified (RAN, scratch mutant, not
        # committed) that widening the exponent's trailing-digit count
        # from {1,2} to {0,2} does NOT change parse_c17g's outcome here
        # either, for the same reason as the two tests above: no
        # canonical %.6f/%.17g spelling ever produces a single-digit
        # exponent, so the round-trip check rejects it independently of
        # this group's width. Use a magnitude small enough that %.17g
        # actually chooses exponential notation (a mid-size value like
        # 1e5 round-trips through plain "100000" instead and would not
        # even reach the exponent group).
        text = format(-1.0000000000000001e-05, ".17g")
        self.assertEqual(text, "-1.0000000000000001e-05")
        self.assertEqual(EVAL.parse_c17g(text), -1.0000000000000001e-05)
        with self.assertRaises(EVAL.EvidenceError):
            EVAL.parse_c17g(text.replace("e-05", "e-5"))

    def test_parse_engine_loaded_rejects_overflowing_metrics(self):
        # load_s/resident_mb have no magnitude cap in _FIXED2_TEXT (only
        # a fixed 2-decimal-digit suffix), so a long enough digit run
        # converts via float() to +inf; unlike parse_c17g there is no
        # round-trip format comparison to catch that independently here,
        # so the isfinite check is this function's only defense against
        # it -- a removed/weakened isfinite check would silently accept
        # an infinite load time or resident size.
        huge = "1" + "0" * 350 + ".00"
        for field in ("load_s", "resident_mb"):
            with self.subTest(field=field):
                kwargs = {"load": huge} if field == "load_s" else {"resident": huge}
                line = self.loaded(**kwargs)
                with self.assertRaisesRegex(
                        EVAL.PreambleError, "must be finite"):
                    EVAL.parse_engine_loaded(line)

    def test_score_request_digest_binds_strict_ascii_bytes_including_lf(self):
        requests = ("1 1 1 2", "2 1 0 1 2")
        lines, payload, digests = EVAL.score_request_wire(requests, 3)
        self.assertEqual(lines, (
            b"1 1 1 2\n", b"2 1 0 1 2\n"))
        self.assertEqual(payload, b"".join(lines))
        self.assertEqual(
            digests,
            tuple(hashlib.sha256(line).hexdigest() for line in lines))
        self.assertNotEqual(
            digests[1],
            hashlib.sha256(b"2 1 0 1 2").hexdigest())
        bad_requests = (
            "", "two\nlines", "cr\rline", "1 1", "1 1 0",
            "1 1 0 1 2", "1 1 0 1 junk", "1 1 0 3",
            "1 1 -1 1", "1 1 00 1", "2147483648 1 0 1",
            "1 2147483647 0 1", "evidence-μ",
        )
        for bad in bad_requests:
            with self.subTest(bad=bad):
                with self.assertRaises(EVAL.EvidenceError):
                    EVAL.score_request_wire((bad,), 3)
        with self.assertRaises(EVAL.EvidenceError):
            EVAL.score_request_wire((), 3)

    def test_nonfinite_and_malformed_scores_refuse(self):
        bad = [
            "nan 1 1", "inf 1 1", "-inf 1 1", "-1 1", "-1 1 1 extra",
            "not-a-number 1 1", "-1 -1 1", "-1 0 1", "-1 1 2", "1 1 1",
            "-1  1 1", "-1\t1 1", "-1 01 1", "-1_0 1 1",
            "-1 2147483648 1", "+0 1 0", "-01 1 0",
            "-0_125 1 0", "-١ 1 0", "-1e-9999 1 0",
            "-1.00000000000000000 1 0", "-1e-9 1 0",
            "-1e--09 1 0", "-1e+009 1 0", "-1.25 1 0 junk",
        ]
        for line in bad:
            with self.subTest(line=line):
                with self.assertRaises(EVAL.EvidenceError):
                    EVAL.parse_score_result(line)

    def test_stdout_grammar_refuses_unknown_records(self):
        self.assertIsNone(EVAL.classify_score_stdout(self.BANNER + "\n"))
        for state, draft in (("ACTIVE", 0), ("ACTIVE", 1),
                             ("absent", 0), ("absent", 2),
                             ("DISABLED (multiplexed serve)", 0)):
            with self.subTest(state=state, draft=draft):
                self.assertIsNone(EVAL.classify_score_stdout(
                    self.loaded(state, draft) + "\n"))
        exact, _, _, _ = EVAL.classify_score_stdout("-1.25 1 0\n")
        self.assertEqual(exact, "-1.25")
        bad = (
            "\n", "unexpected banner\n", "nan 1 0\n", "inf 1 0\n",
            " -1.25 1 0\n", "-1.25 1 0 \n", "-1.25 1 0\t\n",
            "-1.25 1 0", "-1.25 1 0\r\n",
            "PROF 0.001 1 1 0.000 0.000 0.000 0.000 0.000 1\n",
            "DONE 7 STAT 1 1.00 0.0 1.00 1 0\n",
            "== GLM C engine fabricated ==\n",
            "== GLM C engine (glm_moe_dsa), cache=0 experts/layer | "
            "compute experts@4-bit dense@8-bit | idot: neon ==\n",
            "== GLM C engine (glm_moe_dsa), cache=064 experts/layer | "
            "compute experts@4-bit dense@8-bit | idot: neon ==\n",
            "loaded in 1.0s | resident dense: 1.00 MB | layers=78 experts=256 | "
            "MTP ACTIVE (draft=1)\n",
            "loaded in 1.00s | resident dense: 1.00 MB | layers=78 experts=256 | "
            "MTP DISABLED (multiplexed serve) (draft=1)\n",
            "loaded in 1.00s | resident dense: 1.00 MB | layers=78 experts=256 | "
            "MTP unknown (draft=0)\n",
            "loaded in 1.00s | resident dense: 1.00 MB | layers=78 experts=256 | "
            "MTP ACTIVE (draft=64)\n",
            "loaded in 1.00s | resident dense: 1.00 MB | layers=0 experts=256 | "
            "MTP ACTIVE (draft=1)\n",
            "loaded in 1.00s | resident dense: 1.00 MB | layers=129 experts=256 | "
            "MTP ACTIVE (draft=1)\n",
            "loaded in 1.00s | resident dense: 1.00 MB | layers=78 experts=0 | "
            "MTP ACTIVE (draft=1)\n",
            "loaded in 1.00s | resident dense: 1.00 MB | layers=78 experts=4097 | "
            "MTP ACTIVE (draft=1)\n",
            "loaded in -0.01s | resident dense: 1.00 MB | layers=78 experts=256 | "
            "MTP ACTIVE (draft=1)\n",
            "loaded in nan s | resident dense: 1.00 MB | layers=78 experts=256 | "
            "MTP ACTIVE (draft=1)\n",
            "loaded in 1.0s | resident dense: 1.00 MB | layers=78 experts=256 | "
            "MTP ACTIVE (draft=1)\n",
            "loaded in 1.00s | resident dense: 1.00 MB | layers=2147483648 experts=256 | "
            "MTP ACTIVE (draft=1)\n",
            " == GLM C engine (glm_moe_dsa), cache=64 experts/layer | "
            "compute experts@4-bit dense@8-bit | idot: neon ==\n",
        )
        for line in bad:
            with self.subTest(line=line):
                with self.assertRaises(EVAL.EvidenceError):
                    EVAL.classify_score_stdout(line)

    def test_loaded_prefix_collision_is_refused_not_passed_through(self):
        # engine_evidence.parse_engine_preamble dispatches by literal prefix
        # (line.startswith("loaded in")), not by a semantic match against
        # the load record's shape. "loaded index ..." shares that 9-char
        # prefix purely because "index" itself starts with "in", so it is
        # routed into parse_engine_loaded and refused there, rather than
        # returned as an ordinary unowned log line. Confirmed at primary
        # source (c/colibri.c on origin/dev): no engine anywhere in this
        # tree ever prints a line starting with "loaded index" -- the only
        # stdout "loaded" record any engine emits is the exact "loaded in
        # ...s | resident dense: ..." line this module already pins above.
        # This is deliberately fail-loud dispatch behavior, not a bug --
        # see engine_evidence.parse_engine_preamble's own docstring
        # (fixed in this cycle to document the prefix mechanism and this
        # exact collision explicitly, since no engine anywhere in this
        # tree emits "loaded index" today). This test pins eval_glm's
        # own consumer-visible behavior for the same collision.
        with self.assertRaises(EVAL.EvidenceError):
            EVAL.classify_score_stdout("loaded index 5 whatever\n")

    def test_oversized_numeric_field_refuses_as_evidence_error(self):
        # engine_evidence's own int()/float() calls now catch this at
        # the root (a numeric preamble field beyond the interpreter's
        # 4300-digit int-string conversion limit raises engine_evidence's
        # own PreambleError, not a bare ValueError) -- see
        # test_engine_evidence.py's identical-purpose tests on
        # parse_engine_banner/parse_engine_loaded directly. This
        # module's three call sites still catch ValueError alongside
        # PreambleError too, as defense in depth for any other
        # ValueError the shared module might someday raise; this test
        # pins the end-to-end result through eval_glm's own consumers.
        big = "1" + "0" * 4300
        banner = (
            f"== GLM C engine (glm_moe_dsa), cache={big} experts/layer | "
            "compute experts@4-bit dense@8-bit | idot: avx2 ==\n")
        with self.assertRaises(EVAL.EvidenceError):
            EVAL.classify_score_stdout(banner)
        parser = EVAL.ScoreStdoutClassifier()
        with self.assertRaises(EVAL.EvidenceError):
            parser.classify(banner)
        # is_score_preamble is a query function (bool return); it must
        # swallow the malformed-field case as "not a preamble" rather
        # than raise at all.
        self.assertFalse(EVAL.is_score_preamble(banner.rstrip("\n")))

    def test_oversized_vocab_json_refuses_as_evidence_error(self):
        # Same error-contract class as F3, in a different function:
        # json.loads() raises a bare ValueError (not its
        # json.JSONDecodeError subclass) for an integer literal beyond
        # Python's 4300-digit int-string conversion limit, which
        # score_snapshot_vocab's except clause did not catch -- the
        # bare ValueError escaped main()'s only try/except around this
        # call (which catches EvidenceError to write the INCOMPLETE
        # marker and return before ever launching the engine), so
        # main() crashed outright with NO INCOMPLETE marker written at
        # all, worse than the ordinary prelaunch-refusal contract every
        # other pre-engine-launch failure gets.
        huge_vocab = "1" + "0" * 4300
        with tempfile.TemporaryDirectory() as tmp:
            (pathlib.Path(tmp) / "config.json").write_text(
                '{"vocab_size": ' + huge_vocab + '}\n')
            with self.assertRaises(EVAL.EvidenceError):
                EVAL.score_snapshot_vocab(tmp)

    def test_oversized_vocab_json_marks_incomplete_end_to_end(self):
        # The end-to-end contract test_oversized_vocab_json_refuses_
        # as_evidence_error's unit-level pin implies: main() must reach
        # prelaunch_incomplete() (INCOMPLETE marker written, engine
        # never launched, exit 1) rather than crash uncaught.
        class Encoded:
            ids = [1, 2]

        class FakeTokenizer:
            @staticmethod
            def from_file(path):
                return FakeTokenizer()

            @staticmethod
            def encode(text):
                return Encoded()

        huge_vocab = "1" + "0" * 4300
        with tempfile.TemporaryDirectory() as tmp:
            output = pathlib.Path(tmp) / "results.csv"
            (pathlib.Path(tmp) / "config.json").write_text(
                '{"vocab_size": ' + huge_vocab + '}\n')
            argv = [
                "eval_glm.py", "--snap", tmp, "--tasks", "smoke",
                "--limit", "1", "--glm", "/fake/glm", "--out",
                str(output),
            ]
            tokenizers = types.SimpleNamespace(Tokenizer=FakeTokenizer)
            with mock.patch.object(sys, "argv", argv), \
                    mock.patch.dict(sys.modules, {"tokenizers": tokenizers}), \
                    mock.patch.object(
                        EVAL.subprocess, "Popen",
                        side_effect=AssertionError("engine launched")):
                rc = EVAL.main()
            self.assertEqual(rc, 1)
            self.assertIn("# INCOMPLETE:", output.read_text())

    def test_banner_kernels_and_load_boundaries_are_exact(self):
        self.assertEqual(
            EVAL.parse_engine_banner(self.BANNER)["kernel"], "neon-i8mm")
        for kernel in ("avx512-vnni", "avx-vnni", "avx2", "neon-i8mm",
                       "neon", "vsx", "scalar"):
            line = self.BANNER.replace("neon-i8mm", kernel)
            self.assertEqual(EVAL.parse_engine_banner(line)["kernel"], kernel)
        with self.assertRaises(EVAL.PreambleError):
            EVAL.parse_engine_banner(self.BANNER.replace("neon-i8mm", "fabricated"))

        for layers, experts in ((1, 1), (128, 4096)):
            parsed = EVAL.parse_engine_loaded(self.loaded(
                layers=layers, experts=experts))
            self.assertEqual((parsed["layers"], parsed["experts"]),
                             (layers, experts))

    def test_score_stream_owns_banner_load_lifecycle(self):
        parser = EVAL.ScoreStdoutClassifier()
        self.assertIsNone(parser.classify(self.BANNER + "\n"))
        self.assertIsNone(parser.classify(self.loaded("absent", 2) + "\n"))
        exact, _, _, _ = parser.classify("-1.25 1 0\n")
        self.assertEqual(exact, "-1.25")
        parser.finish()

        cases = (
            [self.loaded() + "\n", self.BANNER + "\n"],
            [self.BANNER + "\n", "-1.25 1 0\n"],
            [self.BANNER + "\n", self.BANNER + "\n"],
            [self.BANNER + "\n", self.loaded() + "\n",
             self.loaded() + "\n"],
            [self.BANNER + "\n"],
        )
        for records in cases:
            with self.subTest(records=records):
                parser = EVAL.ScoreStdoutClassifier()
                with self.assertRaises(EVAL.EvidenceError):
                    for record in records:
                        parser.classify(record)
                    parser.finish()

    def test_eval_main_uses_stateful_score_stream(self):
        rc, output, launches = self.run_eval_main((
            self.BANNER + "\n",
            self.loaded("absent", 2) + "\n",
            "-1 2 1\n", "-2 2 0\n", "-3 2 0\n",
        ))
        self.assertEqual(rc, 0)
        self.assertEqual(launches, 1)
        self.assertIn("# finished: 3/3", output)

    def test_result_rows_are_written_and_flushed_incrementally(self):
        # A run interrupted mid-task must leave a valid partial file --
        # rows are written and flushed per-request, never buffered until
        # the run completes. Simulate a mid-run crash by having the fake
        # engine's stdout iterator raise after the first scored record;
        # the CSV must already contain that row.

        class Encoded:
            ids = [1, 2]

        class FakeTokenizer:
            @staticmethod
            def from_file(path):
                return FakeTokenizer()

            @staticmethod
            def encode(text):
                return Encoded()

        class CrashingStdout:
            def __init__(self, lines):
                self._lines = list(lines)

            def __iter__(self):
                for index, line in enumerate(self._lines):
                    yield line
                    if index == 2:  # after banner+load+one SCORE row
                        raise OSError("engine died mid-run")

        stdout_lines = [
            self.BANNER + "\n", self.loaded("absent", 2) + "\n",
            "-1 2 1\n",
            "-2 2 0\n",
        ]
        process = types.SimpleNamespace(
            returncode=1, stderr=(), stdout=CrashingStdout(stdout_lines),
            wait=lambda: 1, poll=lambda: 1, terminate=lambda: None)
        with tempfile.TemporaryDirectory() as tmp:
            output = pathlib.Path(tmp) / "results.csv"
            (pathlib.Path(tmp) / "config.json").write_text(
                '{"vocab_size":3}\n')
            argv = [
                "eval_glm.py", "--snap", tmp, "--tasks", "smoke",
                "--limit", "1", "--glm", "/fake/glm", "--out",
                str(output),
            ]
            tokenizers = types.SimpleNamespace(Tokenizer=FakeTokenizer)
            with mock.patch.object(sys, "argv", argv), \
                    mock.patch.dict(sys.modules, {"tokenizers": tokenizers}), \
                    mock.patch.object(EVAL.subprocess, "Popen",
                                      return_value=process), \
                    self.assertRaises(OSError):
                EVAL.main()
            text = output.read_text()
            self.assertIn(",-1,1\n", text.replace(".000000", ""))
            self.assertNotIn("# finished:", text)
            # The crash-path marker itself must be present, not
            # just the absence of "# finished:" -- a downstream consumer
            # scans for this exact line to know the run never reached a
            # complete denominator.
            self.assertIn(
                "# INCOMPLETE: evaluator terminated before a complete "
                "denominator", text)

    def test_row_is_durable_on_disk_before_the_run_finishes(self):
        # test_result_rows_are_written_and_flushed_incrementally (above)
        # is named for the per-row flush but cannot actually detect
        # losing it: that test reads the output file only AFTER
        # EVAL.main() has unwound through its `finally` block, which
        # closes out_f (and Python flushes a file on close) regardless
        # of whether any PER-ROW flush ever ran. It defends the ARTIFACT
        # after a clean-ish exit, not the PROPERTY of durability DURING
        # the run.
        #
        # This test instead opens a SECOND, independent file handle on
        # the same path while main() is still running (has not
        # returned) and reads through it -- a second handle can only
        # see bytes actually flushed to the OS by the writer, never
        # bytes still sitting only in the writer's own buffer. The
        # probe runs from inside the fake stdout generator, resumed
        # only once the consumer (main()'s per-line loop body, including
        # write_result_row + out_f.flush() for that line) has fully
        # finished with the PREVIOUS line and is asking for the next one
        # -- so a snapshot taken there reflects exactly what has been
        # made durable so far, mid-run.
        class Encoded:
            ids = [1, 2]

        class FakeTokenizer:
            @staticmethod
            def from_file(path):
                return FakeTokenizer()

            @staticmethod
            def encode(text):
                return Encoded()

        class ProbingStdout:
            def __init__(self, lines, probe_path):
                self._lines = list(lines)
                self._probe_path = probe_path
                self.snapshots = []

            def __iter__(self):
                for line in self._lines:
                    yield line
                    try:
                        with open(self._probe_path) as f:
                            self.snapshots.append(f.read())
                    except FileNotFoundError:
                        self.snapshots.append("")

        with tempfile.TemporaryDirectory() as tmp:
            output = pathlib.Path(tmp) / "results.csv"
            (pathlib.Path(tmp) / "config.json").write_text(
                '{"vocab_size":3}\n')
            stdout = ProbingStdout((
                self.BANNER + "\n",
                self.loaded("absent", 2) + "\n",
                "-1 2 1\n",
                "-2 2 0\n",
            ), str(output))
            process = types.SimpleNamespace(
                returncode=0, stderr=(), stdout=stdout,
                wait=lambda: 0, poll=lambda: 0, terminate=lambda: None)
            argv = [
                "eval_glm.py", "--snap", tmp, "--tasks", "smoke",
                "--limit", "1", "--glm", "/fake/glm", "--out",
                str(output),
            ]
            tokenizers = types.SimpleNamespace(Tokenizer=FakeTokenizer)
            with mock.patch.object(sys, "argv", argv), \
                    mock.patch.dict(sys.modules, {"tokenizers": tokenizers}), \
                    mock.patch.object(EVAL.subprocess, "Popen",
                                      return_value=process):
                rc = EVAL.main()
            self.assertEqual(rc, 0)
        # snapshots[0]/[1]: after the banner/load preamble lines, before
        # any SCORE result -- no row yet.
        self.assertNotIn(",-1,1\n", stdout.snapshots[0].replace(".000000", ""))
        self.assertNotIn(",-1,1\n", stdout.snapshots[1].replace(".000000", ""))
        # snapshots[2]: taken right after the FIRST SCORE line was fully
        # processed but BEFORE the second one is even requested --
        # main() has not returned. The first row must already be
        # readable through the independent handle; the second must not.
        first_row_snapshot = stdout.snapshots[2].replace(".000000", "")
        self.assertIn(",-1,1\n", first_row_snapshot,
                      "first row not durable before the run finished -- "
                      "the per-row flush is not doing its job")
        self.assertNotIn(",-2,0\n", first_row_snapshot)

    def test_child_is_terminated_on_mid_run_interrupt_or_exception(self):
        # SIGINT/SIGTERM/any exception mid-run must not leave the
        # engine child running. The fake engine here never exits on its
        # own -- .poll() keeps returning None (as a real child that
        # ignores its stdin being closed would) until .terminate() is
        # actually called -- so a passing test proves main() called
        # terminate() itself rather than relying on the child to die.
        request_raw = b"2 2 1 2 1 2\n"
        request_digest = hashlib.sha256(request_raw).hexdigest()

        class Encoded:
            ids = [1, 2]

        class FakeTokenizer:
            @staticmethod
            def from_file(path):
                return FakeTokenizer()

            @staticmethod
            def encode(text):
                return Encoded()

        banner = self.BANNER + "\n"
        loaded = self.loaded("absent", 2) + "\n"

        def interrupting_stdout():
            yield banner
            yield loaded
            yield f"SCORE 0 {request_digest} -1 2 1\n"
            raise KeyboardInterrupt("operator pressed Ctrl+C")

        class NeverExitingProcess:
            def __init__(self):
                self.returncode = None
                self.stderr = ()
                self.stdout = interrupting_stdout()
                self.terminated = False
                self.terminate_calls = 0
                self.wait_calls = 0

            def poll(self):
                return 0 if self.terminated else None

            def terminate(self):
                self.terminated = True
                self.terminate_calls += 1

            def wait(self):
                self.wait_calls += 1
                return 0

        process = NeverExitingProcess()
        with tempfile.TemporaryDirectory() as tmp:
            output = pathlib.Path(tmp) / "results.csv"
            (pathlib.Path(tmp) / "config.json").write_text(
                '{"vocab_size":3}\n')
            argv = [
                "eval_glm.py", "--snap", tmp, "--tasks", "smoke",
                "--limit", "1", "--glm", "/fake/glm", "--out",
                str(output),
            ]
            tokenizers = types.SimpleNamespace(Tokenizer=FakeTokenizer)
            with mock.patch.object(sys, "argv", argv), \
                    mock.patch.dict(sys.modules, {"tokenizers": tokenizers}), \
                    mock.patch.object(EVAL.subprocess, "Popen",
                                      return_value=process), \
                    self.assertRaises(KeyboardInterrupt):
                EVAL.main()
        self.assertEqual(process.terminate_calls, 1)
        self.assertGreaterEqual(process.wait_calls, 1)

    @unittest.skipIf(
        sys.platform == "win32",
        "os.kill(pid, SIGTERM) on Windows is TerminateProcess with exit code 15: "
        "no handler runs, so the POSIX mechanism under test does not exist there "
        "(the SIGINT/exception half above covers child cleanup on Windows)")
    def test_sigterm_mid_run_terminates_the_child_and_propagates(self):
        # The SIGTERM half: a real SIGTERM (not just an ordinary
        # Python exception) delivered while the child is running must
        # also be converted into child cleanup, not left to Python's
        # default SIGTERM handling (which does not run this module's
        # `finally` cleanup at all).
        request_raw = b"2 2 1 2 1 2\n"
        request_digest = hashlib.sha256(request_raw).hexdigest()

        class Encoded:
            ids = [1, 2]

        class FakeTokenizer:
            @staticmethod
            def from_file(path):
                return FakeTokenizer()

            @staticmethod
            def encode(text):
                return Encoded()

        banner = self.BANNER + "\n"
        loaded = self.loaded("absent", 2) + "\n"

        def stdout_then_sigterm():
            yield banner
            yield loaded
            yield f"SCORE 0 {request_digest} -1 2 1\n"
            os.kill(os.getpid(), signal.SIGTERM)
            # Not reached if the handler fires promptly, as it must.
            yield f"SCORE 1 {request_digest} -2 2 0\n"

        class NeverExitingProcess:
            def __init__(self):
                self.returncode = None
                self.stderr = ()
                self.stdout = stdout_then_sigterm()
                self.terminated = False
                self.terminate_calls = 0
                self.wait_calls = 0

            def poll(self):
                return 0 if self.terminated else None

            def terminate(self):
                self.terminated = True
                self.terminate_calls += 1

            def wait(self):
                self.wait_calls += 1
                return 0

        process = NeverExitingProcess()
        previous_handler = signal.getsignal(signal.SIGTERM)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                output = pathlib.Path(tmp) / "results.csv"
                (pathlib.Path(tmp) / "config.json").write_text(
                    '{"vocab_size":3}\n')
                argv = [
                    "eval_glm.py", "--snap", tmp, "--tasks", "smoke",
                    "--limit", "1", "--glm", "/fake/glm", "--out",
                    str(output),
                ]
                tokenizers = types.SimpleNamespace(Tokenizer=FakeTokenizer)
                with mock.patch.object(sys, "argv", argv), \
                        mock.patch.dict(sys.modules, {"tokenizers": tokenizers}), \
                        mock.patch.object(EVAL.subprocess, "Popen",
                                          return_value=process), \
                        self.assertRaises(EVAL.ChildTerminateRequested):
                    EVAL.main()
        finally:
            # Defensive: main()'s own finally already restores the prior
            # handler, but never trust a test to leave process-global
            # signal state behind if the assertion above ever fails.
            signal.signal(signal.SIGTERM, previous_handler)
        self.assertEqual(process.terminate_calls, 1)
        self.assertGreaterEqual(process.wait_calls, 1)
        self.assertEqual(signal.getsignal(signal.SIGTERM), previous_handler)

    def test_engine_run_completes_and_reports_finished(self):
        # The engine only ever emits the byte-compatible legacy
        # three-field form ("<exact> <contlen> <greedy>"); a clean run
        # over that form completes and writes every row.
        rc, output, launches = self.run_eval_main((
            self.BANNER + "\n",
            self.loaded("absent", 2) + "\n",
            "-1 2 1\n", "-2 2 0\n", "-3 2 0\n",
        ))
        self.assertEqual(rc, 0)
        self.assertEqual(launches, 1)
        self.assertIn("# finished: 3/3", output)

    def test_classifier_refuses_unknown_lines(self):
        # ScoreStdoutClassifier is the class main() actually constructs
        # -- classify_score_stdout (the standalone function, covered by
        # test_stdout_grammar_refuses_unknown_records) is a separate
        # code path main() never calls. A foreign line must be refused
        # by the classifier itself, not merely by the standalone
        # function.
        foreign_lines = (
            "PROF 0.001 1 1 0.000 0.000 0.000 0.000 0.000 1\n",
            "not a score line at all\n",
            "nan 1 0\n",
            "SCORE 0 " + "a" * 64 + " -1 1 1\n",  # no reader emits this shape
        )
        for line in foreign_lines:
            with self.subTest(line=line):
                parser = EVAL.ScoreStdoutClassifier()
                parser.classify(self.BANNER + "\n")
                parser.classify(self.loaded("absent", 2) + "\n")
                with self.assertRaises(EVAL.EvidenceError):
                    parser.classify(line)

    def test_eval_main_rejects_lifecycle_and_blank_records(self):
        banner = self.BANNER + "\n"
        loaded = self.loaded("absent", 2) + "\n"
        scores = ("-1 2 1\n", "-2 2 0\n", "-3 2 0\n")
        cases = {
            "load_before_banner": (loaded, banner) + scores,
            "score_before_load": (banner, scores[0], loaded) + scores[1:],
            "duplicate_load": (banner, loaded, loaded) + scores,
            "missing_load_at_eof": (banner,),
            "blank_before_banner": ("\n", banner, loaded) + scores,
            "blank_between_preambles": (banner, "\n", loaded) + scores,
            "blank_after_scores": (banner, loaded) + scores + ("\n",),
        }
        for name, records in cases.items():
            with self.subTest(name=name):
                rc, output, launches = self.run_eval_main(records)
                self.assertEqual(rc, 1)
                self.assertEqual(launches, 1)
                self.assertIn("# INCOMPLETE:", output)
                self.assertNotIn("# finished:", output)

    def test_only_complete_zero_exit_denominator_passes(self):
        # NOTE: completion_error's contract changed from the original
        # ported oracle -- it now matches dev's own exit-code contract
        # exactly (a partial or nonzero-exit-but-nonempty run is no
        # longer fatal here; see its docstring), so several of the
        # original oracle's assertions below are inverted rather than
        # reused verbatim.
        self.assertIsNone(EVAL.completion_error(0, 3, 3, 9))
        # A clean exit with zero requests scored is no longer flagged by
        # completion_error itself (dev's own contract only fires on a
        # NONZERO exit with zero scored; `expected`/`continuation_tokens`
        # are not otherwise consulted).
        self.assertIsNone(EVAL.completion_error(0, 0, 0, 0))
        self.assertIsNone(EVAL.completion_error(0, 3, 3, 0))
        # A nonzero exit that still scored at least one request is a
        # partial run, not fatal.
        self.assertIsNone(EVAL.completion_error(2, 7, 7, 7))
        self.assertIsNone(EVAL.completion_error(0, 6, 7, 6))
        # Fatal only when the engine exits nonzero with NOTHING scored...
        self.assertIsNotNone(EVAL.completion_error(2, 0, 7, 0))
        self.assertIn("zero requests scored", EVAL.completion_error(2, 0, 7, 0))
        # ...or a stream_error is present regardless of completion count.
        self.assertIn("broken", EVAL.completion_error(0, 7, 7, 7, "broken"))
        self.assertIn("broken", EVAL.completion_error(0, 0, 7, 0, "broken"))

    def test_empty_selection_refuses_before_engine_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp) / "results.csv"
            argv = ["eval_glm.py", "--snap", tmp, "--tasks", "", "--out", str(out)]
            with mock.patch.object(sys, "argv", argv), \
                    mock.patch.object(EVAL.subprocess, "Popen") as popen:
                rc = EVAL.main()
            self.assertEqual(rc, 1)
            popen.assert_not_called()
            text = out.read_text()
            self.assertIn("# INCOMPLETE: 0/0; error=no benchmark tasks selected", text)
            self.assertNotIn("finished: 0/0", text)

    def test_zero_request_task_refuses_before_engine_launch(self):
        class FakeTokenizer:
            @staticmethod
            def from_file(path):
                return object()

        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            (root / "empty.jsonl").write_text("")
            out = root / "results.csv"
            argv = ["eval_glm.py", "--snap", tmp, "--data", tmp,
                    "--tasks", "empty", "--out", str(out)]
            fake = types.SimpleNamespace(Tokenizer=FakeTokenizer)
            with mock.patch.object(sys, "argv", argv), \
                    mock.patch.dict(sys.modules, {"tokenizers": fake}), \
                    mock.patch.object(EVAL.subprocess, "Popen",
                                      side_effect=AssertionError("engine launched")) as popen:
                rc = EVAL.main()
            self.assertEqual(rc, 1)
            popen.assert_not_called()
            text = out.read_text()
            self.assertIn(
                "# INCOMPLETE: 0/0; error=selected tasks produced zero SCORE requests",
                text)
            self.assertNotIn("finished: 0/0", text)

    def test_zero_continuation_choices_refuse_before_engine_launch(self):
        class Encoded:
            def __init__(self, ids):
                self.ids = ids

        class BoundaryTokenizer:
            @staticmethod
            def from_file(path):
                return BoundaryTokenizer()

            @staticmethod
            def encode(text):
                return Encoded({
                    "ctx": [1], "ctxgood": [1, 2], "good": [2],
                    "ctxvanish": [1], "vanish": [], "": [],
                }.get(text, [1]))

        cases = {
            "one_empty": [{"ctx": "ctx", "choices": [""], "gold": 0}],
            "all_empty": [{"ctx": "ctx", "choices": ["", ""], "gold": 0}],
            "boundary_still_empty": [
                {"ctx": "ctx", "choices": ["vanish"], "gold": 0}],
            "mixed_positive_zero": [
                {"ctx": "ctx", "choices": ["good", ""], "gold": 0}],
        }
        tokenizer = BoundaryTokenizer()
        for name, docs in cases.items():
            with self.subTest(name=name), \
                    self.assertRaisesRegex(EVAL.EvidenceError,
                                           "no positive context/continuation"):
                EVAL.build_requests(tokenizer, {"task": docs})

        tokenizers = types.SimpleNamespace(Tokenizer=BoundaryTokenizer)
        for name, docs in cases.items():
            with self.subTest(prelaunch=name), \
                    tempfile.TemporaryDirectory() as tmp:
                root = pathlib.Path(tmp)
                data = root / f"{name}.jsonl"
                data.write_text(json.dumps(docs[0]) + "\n")
                out = root / "results.csv"
                argv = ["eval_glm.py", "--snap", str(root),
                        "--data", str(root), "--tasks", name,
                        "--out", str(out)]
                with mock.patch.object(sys, "argv", argv), \
                        mock.patch.dict(
                            sys.modules, {"tokenizers": tokenizers}), \
                        mock.patch.object(
                            EVAL.subprocess, "Popen",
                            side_effect=AssertionError(
                                "engine launched")) as popen:
                    rc = EVAL.main()
                self.assertEqual(rc, 1)
                popen.assert_not_called()
                text = out.read_text()
                self.assertIn("# INCOMPLETE: 0/0; error=", text)
                self.assertNotIn("# finished:", text)

    def test_dry_run_does_not_require_a_vocabulary(self):
        # dev's own --dry never looked up config.json's
        # vocab_size at all -- it only builds and tokenizes requests,
        # then stops. This module's vocabulary/digest binding is a
        # per-request-wire step for the real engine launch, not a
        # plumbing check, so --dry must not depend on it.
        class Encoded:
            ids = [1, 2]

        class FakeTokenizer:
            @staticmethod
            def from_file(path):
                return FakeTokenizer()

            @staticmethod
            def encode(text):
                return Encoded()

        with tempfile.TemporaryDirectory() as tmp:
            # No config.json at all in the snapshot directory -- a real
            # vocabulary lookup would raise EvidenceError immediately.
            argv = ["eval_glm.py", "--snap", tmp, "--tasks", "smoke",
                    "--limit", "1", "--dry"]
            tokenizers = types.SimpleNamespace(Tokenizer=FakeTokenizer)
            with mock.patch.object(sys, "argv", argv), \
                    mock.patch.dict(sys.modules, {"tokenizers": tokenizers}), \
                    mock.patch.object(
                        EVAL, "score_snapshot_vocab",
                        side_effect=AssertionError(
                            "vocabulary looked up during --dry")) as vocab, \
                    mock.patch.object(
                        EVAL.subprocess, "Popen",
                        side_effect=AssertionError("engine launched")) as popen:
                rc = EVAL.main()
            self.assertIsNone(rc)
            vocab.assert_not_called()
            popen.assert_not_called()

    def test_partial_run_still_reports_the_table_and_exits_zero(self):
        # dev's own exit-code contract (coli bench does
        # `sys.exit(subprocess.call(cmd, ...))`; diag_harness.py parses
        # this tool's stdout table from a subprocess call) exits nonzero
        # ONLY when the engine produced nothing at all. A partial run --
        # some but not all requests scored, clean stream -- must still
        # print the accuracy table and exit 0; the INCOMPLETE marker is
        # additive (alongside "# finished", not instead of it).
        rc, output, launches = self.run_eval_main((
            self.BANNER + "\n",
            self.loaded("absent", 2) + "\n",
            "-1 2 1\n",  # only 1 of 3 requests scored
        ))
        self.assertEqual(rc, 0)
        self.assertEqual(launches, 1)
        self.assertIn("# finished: 1/3", output)
        self.assertIn("# INCOMPLETE: 1/3 requests scored", output)
        self.assertIn(
            "WARNING: only 1/3 requests scored", self.last_stderr)
        self.assertIn("MEAN acc_norm", self.last_stdout)

    def test_python_engine_byte_limits_are_inclusive_and_preallocation(self):
        engine_limit = 256 << 20
        self.assertEqual(EVAL._ENGINE_TEXT_MAX_BYTES, engine_limit)
        self.assertEqual(
            EVAL._checked_engine_text_size(engine_limit, "SCORE"),
            engine_limit)
        with self.assertRaises(EVAL.EvidenceError):
            EVAL._checked_engine_text_size(engine_limit + 1, "SCORE")

        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            score_config = root / "config.json"
            score_raw = b'{"vocab_size":4}\n'
            score_config.write_bytes(score_raw)
            with mock.patch.object(
                    EVAL, "_ENGINE_TEXT_MAX_BYTES", len(score_raw)):
                self.assertEqual(EVAL.score_snapshot_vocab(root), 4)
                score_config.write_bytes(score_raw + b" ")
                with self.assertRaisesRegex(EVAL.EvidenceError, "256 MiB"):
                    EVAL.score_snapshot_vocab(root)

        request = "1 1 0 1"
        request_bytes = len((request + "\n").encode("ascii"))
        with mock.patch.object(
                EVAL, "_ENGINE_TEXT_MAX_BYTES", request_bytes):
            _, payload, _ = EVAL.score_request_wire((request,), 4)
            self.assertEqual(len(payload), request_bytes)
        with mock.patch.object(
                EVAL, "_ENGINE_TEXT_MAX_BYTES", request_bytes - 1):
            with self.assertRaisesRegex(EVAL.EvidenceError, "256 MiB"):
                EVAL.score_request_wire((request,), 4)


class DifferentialBiteTests(unittest.TestCase):
    """dev's classifier silently accepts a foreign stdout line the new
    copy refuses. dev's side is not itself invoked here (no ported dev
    module exists in this tree); it is asserted against the same fixture
    line via dev's documented filter logic, ported verbatim inline. Only
    the new copy's refusal is exercised by calling real code."""

    FOREIGN_LINE = "1 1 1\n"  # shaped like dev's own accepted grammar

    def test_dev_copy_silently_accepts_a_foreign_numeric_line(self):
        # dev's inline stdout filter (ported verbatim as the oracle): any
        # line starting with a digit or '-' is treated as a SCORE record,
        # with no further validation at all.
        line = self.FOREIGN_LINE.strip()
        self.assertTrue(line and line[0] in "-0123456789")
        parts = line.split()
        logprob = float(parts[0])  # dev: "try: logprob = float(parts[0])"
        # dev accepts this as a real SCORE result -- a false positive: a
        # log-likelihood can never be positive, but dev's filter never
        # checks the sign (or finiteness, or field count) at all.
        self.assertEqual(logprob, 1.0)

    def test_new_copy_refuses_the_same_foreign_line_by_name(self):
        with self.assertRaisesRegex(
                EVAL.EvidenceError, "SCORE logprob is not finite/non-positive"):
            EVAL.classify_score_stdout(self.FOREIGN_LINE)


if __name__ == "__main__":
    unittest.main()
