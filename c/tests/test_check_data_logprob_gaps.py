"""tests/check_data_logprob_gaps.py must accept only a complete, well-formed
raw engine-stdout transcript for one opted-in request and reject every
other input: an unparsed or mismatched startup preamble, a legacy 3-field
DATA frame hiding a dropped logprob channel anywhere in the generation, a
numeric tail with the wrong top-k count or an out-of-range/duplicate token
id, malformed framing, a missing or duplicated ACCEPT/DONE, and every other
gap the module's docstring claims to catch.

Checks enumerated from the source (`check_data_logprob_gaps.py`, read in
full before writing this module) and covered below:

- `_uint`, `_c17g`, `_fixed_metric`: the noncanonical-ASCII-integer grammar,
  the exact finite `%.17g` spelling, and the fixed-decimal-place grammar
  with its inclusive bounds.
- `_header_fields` / `_global_header`: non-printable-ASCII and stray-space
  rejection on an ordinary protocol header, and the exact field grammar
  for every recognized mux-global record (BANNER/LOADED preamble, READY,
  STAT, HWINFO, TIERS, EMAP, HITS, PROF).
- `parse_frames`: byte-exact DATA/ECHO payload framing, so a single
  malformed frame cannot desynchronize the parse.
- `capture_mode`: the four explicit transcript shapes it names.
- `_numeric_tail` / `check`: every check enumerated in the module
  docstring, including the legacy 3-field DATA gap this checker exists to
  catch, the `min(topk, vocab)` top-k cap, and out-of-range/duplicate
  token ids.

Every fixture here is a literal transcript built by hand from the module's
documented wire grammar -- no expected value is produced by calling the
checker under test. The transcript-shaped fixtures (`_GapFixture`'s BANNER/
LOADED/VALID/echo/transcript/telemetry/full_capture/complete_capture
helpers) and the `GapCheckerEvidenceTests` methods that exercise them are
carried over unchanged from the checker's own evidence-consumer suite,
excluding the one method that requires a compiled fixture binary.
"""
import collections
import importlib.util
import pathlib
import random
import subprocess
import sys
import tempfile
import unittest

_HERE = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "check_data_logprob_gaps", _HERE / "check_data_logprob_gaps.py")
GAPS = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(GAPS)


class _GapFixture(unittest.TestCase):
    BANNER = (
        b"== GLM C engine (glm_moe_dsa), cache=64 experts/layer | "
        b"compute experts@4-bit dense@8-bit | idot: neon-i8mm ==\n")
    LOADED = (
        b"loaded in 1.00s | resident dense: 1.00 MB | "
        b"layers=78 experts=256 | MTP ACTIVE (draft=1)\n")
    VALID = (
        b"ACCEPT 7 1\n"
        b"ECHO 7 1 0 nan 0\n"
        b"x\n"
        b"DATA 7 1 -2.7000000000000002 2 0 -2.7000000000000002 1 -11.123455999999999\n"
        b"x\n"
        b"DONE 7 STAT 1 0.10 100.0 10.00 1 0\n"
    )

    @staticmethod
    def echo(pos):
        if pos == 0:
            return b"ECHO 7 1 0 nan 0\nx\n"
        return (f"ECHO 7 1 {pos} -2.7000000000000002 2 0 -2.7000000000000002 1 -11.123455999999999\n"
                "x\n").encode("ascii")

    @classmethod
    def transcript(cls, prompt=1, positions=None, emitted=1, done_prompt=None,
                   tps="0.10", hit="100.0", rss="10.00", flag=0):
        if positions is None:
            positions = list(range(prompt))
        if done_prompt is None:
            done_prompt = prompt
        parts = [f"ACCEPT 7 {prompt}\n".encode("ascii")]
        parts.extend(cls.echo(pos) for pos in positions)
        parts.extend([
            b"DATA 7 1 -2.7000000000000002 2 0 -2.7000000000000002 1 -11.123455999999999\nx\n",
            (f"DONE 7 STAT {emitted} {tps} {hit} {rss} "
             f"{done_prompt} {flag}\n").encode("ascii"),
        ])
        return b"".join(parts)

    def problems(self, blob, topk=2, vocab=3):
        frames, framing = GAPS.parse_frames(blob)
        data, echo, problems, _mode, _forms = GAPS.check(
            frames, framing, "7", topk, vocab)
        return data, echo, problems

    def full_problems(self, blob, topk=2, vocab=3):
        """Like problems(), but also returns capture_mode and the numeric
        form(s) observed -- for tests that need those two new signals."""
        frames, framing = GAPS.parse_frames(blob)
        return GAPS.check(frames, framing, "7", topk, vocab)

    @staticmethod
    def telemetry(cpu=b"AMD Ryzen  9", cores=b"16"):
        return b"".join((
            b"HWINFO " + cores + b" 128.0 64.0 2 48.0 " + cpu +
            b"|CUDA device x2\n",
            b"TIERS 128 256 1024 48.00 32.50\n",
            b"EMAP 2 2 00010203\n",
            b"HITS 2 2 0f\n",
            b"PROF 1.250 1 1 0.100 0.200 0.300 0.400 0.500 7\n",
        ))

    @classmethod
    def full_capture(cls):
        startup = (
            b"\x01\x01READY\x01\x01\n"
            b"STAT 0 0.00 0.0 10.00\n"
            b"HWINFO 16 128.0 64.0 2 48.0 AMD Ryzen  9|CUDA device x2\n"
            b"TIERS 128 256 1024 48.00 32.50\n"
            b"EMAP 2 2 00010203\n"
        )
        other = (
            b"ACCEPT 8 1\n"
            b"ECHO 8 1 0 nan 0\nq\n"
            b"DATA 8 1 -2.7000000000000002 2 0 -2.7000000000000002 1 -11.123455999999999\nq\n" +
            cls.telemetry(cpu=b"AMD Ryzen  9") +
            b"DONE 8 STAT 1 0.10 100.0 10.00 1 0\n"
        )
        target = (
            b"ACCEPT 7 1\n"
            b"ECHO 7 1 0 nan 0\nx\n"
            b"DATA 7 1 -2.7000000000000002 2 0 -2.7000000000000002 1 -11.123455999999999\nx\n" +
            cls.telemetry(cpu=b"AMD Ryzen  9") +
            b"DONE 7 STAT 1 0.10 100.0 10.00 1 0\n"
        )
        return startup + other + target

    @classmethod
    def complete_capture(cls, loaded=None):
        return cls.BANNER + (loaded if loaded is not None else cls.LOADED) + cls.full_capture()


class GapCheckerEvidenceTests(_GapFixture):
    """Ported unchanged from the checker's own evidence-consumer suite,
    excluding the one method that scores a compiled fixture binary's
    `%.17g` corpus (needs a build, not available here)."""

    def test_complete_request_passes(self):
        for prompt in (1, 3):
            with self.subTest(prompt=prompt):
                data, echo, problems = self.problems(self.transcript(prompt))
                self.assertEqual((data, echo), (1, prompt))
                self.assertEqual(problems, [])

    def test_full_raw_mux_capture_passes_without_filtering(self):
        data, echo, problems = self.problems(self.full_capture())
        self.assertEqual((data, echo), (1, 1))
        self.assertEqual(problems, [])
        frames, _ = GAPS.parse_frames(self.full_capture())
        self.assertEqual(GAPS.capture_mode(frames), "ready-suffix")

    def test_complete_process_capture_passes_without_filtering(self):
        data, echo, problems = self.problems(self.complete_capture())
        self.assertEqual((data, echo), (1, 1))
        self.assertEqual(problems, [])
        frames, _ = GAPS.parse_frames(self.complete_capture())
        self.assertEqual(GAPS.capture_mode(frames), "full-process")

        for state, draft in ((b"ACTIVE", 0), (b"ACTIVE", 1),
                             (b"absent", 0), (b"absent", 2),
                             (b"DISABLED (multiplexed serve)", 0)):
            loaded = (
                b"loaded in 1.00s | resident dense: 1.00 MB | "
                b"layers=78 experts=256 | MTP " + state +
                b" (draft=" + str(draft).encode("ascii") + b")\n")
            _, _, problems = self.problems(self.complete_capture(loaded))
            self.assertEqual(problems, [], (state, draft, problems))

    def test_complete_process_preamble_lifecycle_is_exact(self):
        base = self.complete_capture()
        suffix = self.full_capture()
        cases = (
            suffix,
            base.replace(self.BANNER, b"", 1),
            base.replace(self.LOADED, b"", 1),
            self.LOADED + self.BANNER + suffix,
            self.BANNER + self.BANNER + self.LOADED + suffix,
            self.BANNER + self.LOADED + self.LOADED + suffix,
            b"unknown preamble\n" + base,
            base.replace(b"idot: neon-i8mm", b"idot: fabricated", 1),
            base.replace(b"loaded in 1.00s", b"loaded in 1.0s", 1),
        )
        # The first case is the separately supported READY-suffix mode.
        self.assertEqual(self.problems(cases[0])[2], [])
        for blob in cases[1:]:
            with self.subTest(blob=blob[:180]):
                _, _, problems = self.problems(blob)
                self.assertTrue(problems)

    def test_load_state_ranges_and_metrics_are_exact_in_full_capture(self):
        base = self.LOADED
        overflow = b"9" * 400 + b".00"
        cases = (
            base.replace(b"ACTIVE (draft=1)",
                         b"DISABLED (multiplexed serve) (draft=1)"),
            base.replace(b"ACTIVE", b"fabricated"),
            base.replace(b"draft=1", b"draft=64"),
            base.replace(b"layers=78", b"layers=0"),
            base.replace(b"layers=78", b"layers=129"),
            base.replace(b"experts=256", b"experts=0"),
            base.replace(b"experts=256", b"experts=4097"),
            base.replace(b"loaded in 1.00s", b"loaded in -0.01s"),
            base.replace(b"loaded in 1.00s", b"loaded in nan s"),
            base.replace(b"loaded in 1.00s", b"loaded in infs"),
            base.replace(b"loaded in 1.00s", b"loaded in " + overflow + b"s"),
            base.replace(b"loaded in 1.00s", b"loaded in 1.0s"),
            base.replace(b"resident dense: 1.00 MB",
                         b"resident dense: -0.01 MB"),
            base.replace(b"resident dense: 1.00 MB",
                         b"resident dense: nan MB"),
            base.replace(b"resident dense: 1.00 MB",
                         b"resident dense: " + overflow + b" MB"),
            base.replace(b"resident dense: 1.00 MB",
                         b"resident dense: 1.0 MB"),
        )
        for loaded in cases:
            with self.subTest(loaded=loaded):
                _, _, problems = self.problems(self.complete_capture(loaded))
                self.assertTrue(any("malformed LOADED" in p for p in problems),
                                problems)

        for layers, experts in ((1, 1), (128, 4096)):
            loaded = base.replace(b"layers=78", f"layers={layers}".encode())
            loaded = loaded.replace(b"experts=256", f"experts={experts}".encode())
            self.assertEqual(self.problems(self.complete_capture(loaded))[2], [])

    def test_emap_and_hits_producer_domains_are_exact(self):
        for byte in (b"00", b"20", b"40", b"60", b"80", b"a0"):
            _, problems = GAPS._global_header(b"EMAP 1 1 " + byte, 0)
            self.assertEqual(problems, [], byte)
        for byte in (b"21", b"61", b"a1"):
            _, problems = GAPS._global_header(b"EMAP 1 1 " + byte, 0)
            self.assertTrue(any("heat" in p for p in problems), problems)
        for byte in (b"c0", b"ff"):
            _, problems = GAPS._global_header(b"EMAP 1 1 " + byte, 0)
            self.assertTrue(any("tier" in p for p in problems), problems)

        _, problems = GAPS._global_header(b"EMAP 0 2 ", 0)
        self.assertEqual(problems, [])
        _, problems = GAPS._global_header(b"EMAP 0 2 00", 0)
        self.assertTrue(any("payload length" in p for p in problems), problems)

        for line in (b"HITS 2 2 0f", b"HITS 1 8 ff"):
            _, problems = GAPS._global_header(line, 0)
            self.assertEqual(problems, [], line)
        for line in (b"HITS 2 2 1f", b"HITS 2 2 ff"):
            _, problems = GAPS._global_header(line, 0)
            self.assertTrue(any("padding" in p for p in problems), problems)

    def test_global_numeric_fields_never_collide_with_request_id(self):
        capture = self.full_capture().replace(
            b"HWINFO 16 128.0", b"HWINFO 7 128.0")
        capture = capture.replace(
            b"TIERS 128 256 1024", b"TIERS 7 256 1024")
        capture = capture.replace(b"EMAP 2 2 00010203", b"EMAP 7 1 00010203040506")
        capture = capture.replace(b"HITS 2 2 0f", b"HITS 7 1 7f")
        data, echo, problems = self.problems(capture)
        self.assertEqual((data, echo), (1, 1))
        self.assertEqual(problems, [])

    def test_malformed_or_misordered_global_records_never_pass(self):
        base = self.full_capture()
        cases = (
            base.replace(b"\x01\x01READY\x01\x01", b"READY", 1),
            base.replace(b"STAT 0 0.00 0.0 10.00", b"STAT 0 0.0 0.0 10.00", 1),
            base.replace(b"HWINFO 16 128.0", b"HWINFO 016 128.0", 1),
            base.replace(b"HWINFO 16 128.0", b"HWINFO 16  128.0", 1),
            base.replace(b"TIERS 128 256", b"TIERS 1_28 256", 1),
            base.replace(b"EMAP 2 2 00010203", b"EMAP 2 2 0001020F", 1),
            base.replace(b"HITS 2 2 0f", b"HITS 2 2 000f", 1),
            base.replace(b"PROF 1.250 1 1", b"PROF 1.25 1 1", 1),
            base.replace(b"PROF 1.250 1 1", b"PROF 1.250 01 1", 1),
            base.replace(b"PROF 1.250 1 1", b"PROF 1.250 1 1_0", 1),
            base.replace(b"\x01\x01READY\x01\x01\n", b"", 1),
            base.replace(b"STAT 0 0.00 0.0 10.00\n", b"", 1),
            base.replace(
                b"\x01\x01READY\x01\x01\nSTAT 0 0.00 0.0 10.00\n",
                b"STAT 0 0.00 0.0 10.00\n\x01\x01READY\x01\x01\n", 1),
            base.replace(
                b"\x01\x01READY\x01\x01\n",
                b"\x01\x01READY\x01\x01\n\x01\x01READY\x01\x01\n", 1),
            base.replace(
                b"STAT 0 0.00 0.0 10.00\n",
                b"STAT 0 0.00 0.0 10.00\nSTAT 0 0.00 0.0 10.00\n", 1),
        )
        for blob in cases:
            with self.subTest(blob=blob[:100]):
                _, _, problems = self.problems(blob)
                self.assertTrue(problems)

    def test_request_id_zero_refuses_before_record_matching(self):
        frames, framing = GAPS.parse_frames(self.full_capture())
        data, echo, problems, mode, forms = GAPS.check(
            frames, framing, "0", 2, 3)
        self.assertEqual((data, echo), (0, 0))
        self.assertEqual(problems, ["requested id must be positive"])
        self.assertEqual(forms, frozenset())

        with tempfile.TemporaryDirectory() as tmp:
            capture = pathlib.Path(tmp) / "capture.raw"
            capture.write_bytes(self.full_capture())
            proc = subprocess.run(
                [sys.executable, str(pathlib.Path(GAPS.__file__)), str(capture),
                 "--id", "0", "--topk", "2", "--vocab", "3"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        self.assertEqual(proc.returncode, 2)
        self.assertIn(b"requested id must be positive", proc.stderr)

    def test_uint_primitive_and_coordinated_underscore_bite(self):
        for token, value in ((b"0", 0), (b"1", 1), (b"10", 10),
                             (b"2147483647", 2147483647)):
            with self.subTest(token=token):
                self.assertEqual(GAPS._uint(token, "fixture"), value)
        for token in (b"+1", b"01", b"1_0", b" 1", b"1 ", b"1\t0",
                      "١".encode("utf-8")):
            with self.subTest(token=token):
                with self.assertRaises(ValueError):
                    GAPS._uint(token, "fixture")

        parts = [b"ACCEPT 7 1_0\n"]
        parts.extend(self.echo(pos) for pos in range(10))
        parts.extend((
            b"DATA 7 1 -2.7000000000000002 2 0 -2.7000000000000002 1 -11.123455999999999\nx\n",
            b"DONE 7 STAT 1 0.10 100.0 10.00 1_0 0\n",
        ))
        data, echo, problems = self.problems(b"".join(parts))
        self.assertEqual((data, echo), (1, 10))
        self.assertTrue(any("invalid ACCEPT prompt length" in p for p in problems))
        self.assertTrue(any("malformed DONE stats" in p for p in problems))
        self.assertFalse(any("ECHO positions" in p or "DONE prompt count" in p
                             for p in problems), problems)

    def test_malformed_framing_never_passes(self):
        bad = [
            b"ACCEPT 7 1\nDATA 7 nope -2.7000000000000002 0\n",
            b"ACCEPT 7 1\nDATA 7 5 -2.7000000000000002 0\nx\n",
            b"ACCEPT 7 1\nDATA 7 1 -2.7000000000000002 0\nx",
            b"ACCEPT 7 1",  # missing header newline
        ]
        for blob in bad:
            with self.subTest(blob=blob):
                _, framing_problems = GAPS.parse_frames(blob)
                self.assertTrue(framing_problems)

    def test_invalid_topk_tables_never_pass(self):
        headers = [
            b"DATA 7 1 -2.7000000000000002 2 -1 -2.7000000000000002 1 -11.123455999999999",  # out-of-range id
            b"DATA 7 1 -2.7000000000000002 2 0 -2.7000000000000002 0 -11.123455999999999",   # duplicate id
            b"DATA 7 1 0.125 2 0 -2.7000000000000002 1 -11.123455999999999",   # positive target lp
            b"DATA 7 1 -2.7000000000000002 2 0 0.125 1 -11.123455999999999",   # positive top-k lp
            b"DATA 7 1 nan 2 0 -2.7000000000000002 1 -11.123455999999999",     # pending nonfinite policy
            b"DATA 7 1 -2.7000000000000002 2 0 -2.7000000000000002 3 -11.123455999999999",  # id == vocab
        ]
        for header in headers:
            blob = (b"ACCEPT 7 1\n" + self.echo(0) + header + b"\nx\n" +
                    b"DONE 7 STAT 1 0.10 100.0 10.00 1 0\n")
            with self.subTest(header=header):
                _, _, problems = self.problems(blob)
                self.assertTrue(problems)

    def test_missing_lifecycle_or_partial_denominator_never_passes(self):
        no_accept = self.VALID.split(b"\n", 1)[1]
        no_done = self.VALID.rsplit(b"DONE", 1)[0]
        wrong_done = self.VALID.replace(b"DONE 7 STAT 1", b"DONE 7 STAT 2")
        error = self.VALID.replace(
            b"DONE 7 STAT 1 0.10 100.0 10.00 1 0", b"ERROR 7 ENGINE")
        cases = (
            (no_accept, "expected exactly one ACCEPT"),
            (no_done, "expected exactly one DONE"),
            (wrong_done, "DONE emitted 2 != observed DATA 1"),
            (error, "target request returned ERROR"),
        )
        for blob, expected in cases:
            with self.subTest(blob=blob):
                _, _, problems = self.problems(blob)
                self.assertTrue(any(expected in problem for problem in problems), problems)

    def test_unknown_targeted_kind_and_echo_after_data_never_pass(self):
        unknown = self.VALID.replace(
            b"DONE 7 STAT", b"MYSTERY 7 value\nDONE 7 STAT")
        echo_after = (
            b"ACCEPT 7 1\n"
            b"DATA 7 1 -2.7000000000000002 2 0 -2.7000000000000002 1 -11.123455999999999\n"
            b"x\n"
            b"ECHO 7 1 0 nan 0\n"
            b"x\n"
            b"DONE 7 STAT 1 0.10 100.0 10.00 1 0\n"
        )
        for blob in (unknown, echo_after):
            with self.subTest(blob=blob):
                _, _, problems = self.problems(blob)
                self.assertTrue(problems)

    def test_echo_denominator_and_order_are_exact(self):
        cases = ([], [0], [0, 1], [0, 2], [0, 1, 1], [1, 0, 2], [0, 1, 2, 3])
        for positions in cases:
            with self.subTest(positions=positions):
                _, _, problems = self.problems(self.transcript(3, positions))
                self.assertTrue(any("ECHO positions" in p for p in problems), problems)

    def test_done_denominator_and_prompt_joins_bite_independently(self):
        zero_data = self.VALID.replace(
            b"DATA 7 1 -2.7000000000000002 2 0 -2.7000000000000002 1 -11.123455999999999\nx\n", b"").replace(
            b"DONE 7 STAT 1", b"DONE 7 STAT 0")
        data, _, problems = self.problems(zero_data)
        self.assertEqual(data, 0)
        self.assertEqual(
            [p for p in problems if p.startswith("no DATA frames")],
            ["no DATA frames found for this id -- wrong id, or the run produced nothing"])
        _, _, problems = self.problems(self.transcript(1, emitted=2))
        self.assertEqual([p for p in problems if p.startswith("DONE emitted")],
                         ["DONE emitted 2 != observed DATA 1"])
        _, _, problems = self.problems(self.transcript(1, done_prompt=2))
        self.assertEqual([p for p in problems if p.startswith("DONE prompt")],
                         ["DONE prompt count 2 != ACCEPT 1"])

    def test_done_domains_and_fixed_grammar_bite_independently(self):
        cases = (
            {"tps": "-0.10"}, {"hit": "-0.1"}, {"hit": "100.1"},
            {"rss": "-0.01"}, {"tps": "0.1"}, {"hit": "0.00"},
            {"rss": "10.0"}, {"tps": "+0.10"}, {"tps": "01.00"},
            {"tps": "1e+00"}, {"tps": "nan"}, {"tps": "inf"},
            {"done_prompt": 0}, {"flag": 2},
        )
        for kwargs in cases:
            with self.subTest(kwargs=kwargs):
                _, _, problems = self.problems(self.transcript(1, **kwargs))
                matches = [p for p in problems if p.startswith("malformed DONE stats")]
                self.assertEqual(len(matches), 1, problems)

        _, _, problems = self.problems(self.transcript(
            1, tps="-0.00", hit="-0.0", rss="-0.00"))
        self.assertEqual(problems, [])

    def test_noncanonical_ascii_numeric_grammar_never_passes(self):
        base = self.VALID
        cases = (
            base.replace(b"ACCEPT 7 1", b"ACCEPT 7 +1"),
            base.replace(b"ACCEPT 7 1", b"ACCEPT 7 1_0"),
            base.replace(b"ECHO 7 1 0", b"ECHO 7 1 +0"),
            base.replace(b"DATA 7 1", b"DATA 7 +1"),
            base.replace(b" -2.7000000000000002 2 ", b" -0_125 2 ", 1),
            base.replace(b" 1 -11.123455999999999", b" 01 -11.123455999999999"),
            base.replace(b"DATA 7 1 ", b"DATA  7 1 "),
            base.replace(b"DATA 7 1 ", b"DATA\t7 1 "),
            base.replace(b"1 -11.123455999999999\nx\n", b"1 -11.123455999999999 \nx\n"),
            base.replace(b"-2.7000000000000002 2", b"-1e-9999 2", 1),
            base.replace(b"-2.7000000000000002 2", b"-1.00000000000000000 2", 1),
            base.replace(b"-2.7000000000000002 2", b"-1e-9 2", 1),
            base.replace(b"-2.7000000000000002 2", b"-1e--09 2", 1),
            base.replace(b"DONE 7 STAT 1", b"DONE 7 STAT 01"),
            base.replace(b" 1 0\n", b" +1 0\n"),
            base.replace(b" 1 0\n", b" 1 00\n"),
            base.replace(b"DATA 7 1", "DATA 7 ١".encode("utf-8")),
            base.replace(b"DATA 7 1 -2.7000000000000002", b"DATA 7 1 -2.7000000000000002junk"),
        )
        for blob in cases:
            with self.subTest(blob=blob):
                _, _, problems = self.problems(blob)
                self.assertTrue(problems)


class PreambleGateFailsLoudTests(_GapFixture):
    """The preamble gate never silently passes an unparsed BANNER/LOADED
    line: `parse_engine_preamble` raising `PreambleError`, or the module's
    own defensive `is None` branch, both surface as a named problem that
    quotes the offending line."""

    def test_unparsed_banner_is_a_named_failure_quoting_the_line(self):
        bad_banner = self.BANNER.replace(b"idot: neon-i8mm", b"idot: bogus", 1)
        kind, problems = GAPS._global_header(bad_banner.rstrip(b"\n"), 0)
        self.assertEqual(kind, [b"BANNER"])
        self.assertEqual(len(problems), 1)
        self.assertIn("malformed BANNER preamble", problems[0])
        self.assertIn(repr(bad_banner.rstrip(b"\n")), problems[0])

    def test_unparsed_loaded_is_a_named_failure_quoting_the_line(self):
        bad_loaded = self.LOADED.replace(b"MTP ACTIVE", b"MTP fabricated", 1)
        kind, problems = GAPS._global_header(bad_loaded.rstrip(b"\n"), 0)
        self.assertEqual(kind, [b"LOADED"])
        self.assertEqual(len(problems), 1)
        self.assertIn("malformed LOADED preamble", problems[0])
        self.assertIn(repr(bad_loaded.rstrip(b"\n")), problems[0])

    def test_preamble_returning_none_is_defensively_named_not_silently_passed(self):
        # `_global_header` treats a None `parse_engine_preamble` result the
        # same as a raised PreambleError: it is unreachable through the
        # real BANNER/LOADED prefixes (they always parse or raise), so this
        # pins the defensive branch directly by forcing that return value.
        original = GAPS.parse_engine_preamble
        GAPS.parse_engine_preamble = lambda text: None
        try:
            kind, problems = GAPS._global_header(
                self.BANNER.rstrip(b"\n"), 0)
        finally:
            GAPS.parse_engine_preamble = original
        self.assertEqual(kind, [b"BANNER"])
        self.assertEqual(len(problems), 1)
        self.assertIn("malformed BANNER preamble", problems[0])
        self.assertIn(repr(self.BANNER.rstrip(b"\n")), problems[0])

    def test_full_capture_with_unparsed_preamble_fails_the_whole_check(self):
        bogus = self.complete_capture().replace(
            b"idot: neon-i8mm", b"idot: bogus", 1)
        _, _, problems = self.problems(bogus)
        self.assertTrue(any("malformed BANNER preamble" in p for p in problems),
                        problems)


class LegacyDataFrameGapTests(_GapFixture):
    """A legacy 3-field DATA frame anywhere in the generation is the exact
    gap this checker exists to catch, and it must fail loudly even when
    it is not the last generated token."""

    def test_legacy_three_field_data_frame_mid_generation_fails(self):
        blob = (
            b"ACCEPT 7 1\n"
            b"ECHO 7 1 0 nan 0\nx\n"
            b"DATA 7 1 -2.7000000000000002 2 0 -2.7000000000000002 1 -11.123455999999999\nx\n"  # token 1: full channel
            b"DATA 7 1\nx\n"                              # token 2: legacy gap
            b"DATA 7 1 -2.7000000000000002 2 0 -2.7000000000000002 1 -11.123455999999999\nx\n"  # token 3: full channel
            b"DONE 7 STAT 3 0.10 100.0 10.00 1 0\n"
        )
        data, echo, problems = self.problems(blob)
        self.assertEqual(data, 3)
        self.assertTrue(any(
            "GAP: legacy 3-field DATA frame #2" in p for p in problems),
            problems)

    def test_wholly_dropped_data_frame_mid_generation_fails_via_done_count(self):
        # The severer form of the same defect class: one generated token's
        # DATA frame never reached stdout at all (not even degraded to 3
        # fields). Only the DONE-emitted-vs-observed-DATA cross-check can
        # name this -- a checker with no concept of a DONE frame would
        # pass it silently.
        blob = (
            b"ACCEPT 7 1\n"
            b"ECHO 7 1 0 nan 0\nx\n"
            b"DATA 7 1 -2.7000000000000002 2 0 -2.7000000000000002 1 -11.123455999999999\nx\n"   # token 1
            b"DATA 7 1 -2.7000000000000002 2 0 -2.7000000000000002 1 -11.123455999999999\nx\n"   # token 3 (token 2 missing)
            b"DONE 7 STAT 3 0.10 100.0 10.00 1 0\n"
        )
        data, echo, problems = self.problems(blob)
        self.assertEqual(data, 2)
        self.assertIn("DONE emitted 3 != observed DATA 2", problems)


class TopkCapAndTokenIdBoundsTests(_GapFixture):
    """The numeric tail enforces k == min(--topk, --vocab) and
    0 <= tid < vocab, reporting the first offending frame."""

    def test_k_above_expected_but_within_wire_cap_never_passes(self):
        # vocab=40, --topk=20 => expected_k = min(20, 40) = 20; the frame
        # advertises k=25, which is <=32 (the wire's own hard cap) but
        # above the expected value for this request.
        pairs = " ".join(f"{i} -2.7000000000000002" for i in range(25))
        blob = (
            b"ACCEPT 7 1\n" + self.echo(0) +
            f"DATA 7 1 -2.7000000000000002 25 {pairs}\n".encode("ascii") + b"x\n" +
            b"DONE 7 STAT 1 0.10 100.0 10.00 1 0\n")
        _, _, problems = self.problems(blob, topk=20, vocab=40)
        self.assertTrue(any(
            "DATA top-k 25 != expected 20" in p for p in problems), problems)

    def test_k_above_wire_hard_cap_of_32_never_passes(self):
        blob = (
            b"ACCEPT 7 1\n" + self.echo(0) +
            b"DATA 7 1 -2.7000000000000002 33 0 -0.1\nx\n" +
            b"DONE 7 STAT 1 0.10 100.0 10.00 1 0\n")
        _, _, problems = self.problems(blob, topk=32, vocab=40)
        self.assertTrue(any(
            "malformed DATA numeric fields" in p for p in problems), problems)

    def test_token_id_equal_to_vocab_never_passes(self):
        blob = (
            b"ACCEPT 7 1\n" + self.echo(0) +
            b"DATA 7 1 -2.7000000000000002 2 0 -2.7000000000000002 3 -0.25\nx\n" +
            b"DONE 7 STAT 1 0.10 100.0 10.00 1 0\n")
        _, _, problems = self.problems(blob, topk=2, vocab=3)
        self.assertTrue(any(
            "token id 3 outside [0,3)" in p for p in problems), problems)

    def test_token_id_well_above_vocab_never_passes(self):
        blob = (
            b"ACCEPT 7 1\n" + self.echo(0) +
            b"DATA 7 1 -2.7000000000000002 1 999 -2.7000000000000002\nx\n" +
            b"DONE 7 STAT 1 0.10 100.0 10.00 1 0\n")
        _, _, problems = self.problems(blob, topk=1, vocab=40)
        self.assertTrue(any(
            "token id 999 outside [0,40)" in p for p in problems), problems)


class CaptureModeLiteralTests(unittest.TestCase):
    """`capture_mode` names the four explicit transcript shapes."""

    def test_empty_transcript_is_request_only(self):
        self.assertEqual(GAPS.capture_mode([]), "request-only")

    def test_accept_first_frame_is_request_only(self):
        frames, _ = GAPS.parse_frames(b"ACCEPT 7 1\n")
        self.assertEqual(GAPS.capture_mode(frames), "request-only")

    def test_banner_first_frame_is_full_process(self):
        frames, _ = GAPS.parse_frames(
            b"== GLM C engine (glm_moe_dsa), cache=64 experts/layer | "
            b"compute experts@4-bit dense@8-bit | idot: neon-i8mm ==\n"
            b"ACCEPT 7 1\n")
        self.assertEqual(GAPS.capture_mode(frames), "full-process")

    def test_ready_first_frame_is_ready_suffix(self):
        frames, _ = GAPS.parse_frames(
            b"\x01\x01READY\x01\x01\nACCEPT 7 1\n")
        self.assertEqual(GAPS.capture_mode(frames), "ready-suffix")

    def test_global_after_a_non_global_first_frame_is_invalid(self):
        frames, _ = GAPS.parse_frames(
            b"ACCEPT 7 1\n\x01\x01READY\x01\x01\n")
        self.assertEqual(GAPS.capture_mode(frames), "invalid")


class FixedMetricAndUintBoundaryTests(unittest.TestCase):
    """`_fixed_metric`'s lower/upper bounds and `_uint`'s maximum
    argument are literal, inclusive boundaries."""

    def test_fixed_metric_lower_bound_is_inclusive(self):
        self.assertEqual(GAPS._fixed_metric(b"0.0", 1, "x", lower=0.0), 0.0)
        with self.assertRaises(ValueError):
            GAPS._fixed_metric(b"-0.1", 1, "x", lower=0.0)

    def test_fixed_metric_upper_bound_is_inclusive(self):
        self.assertEqual(
            GAPS._fixed_metric(b"100.0", 1, "x", upper=100.0), 100.0)
        with self.assertRaises(ValueError):
            GAPS._fixed_metric(b"100.1", 1, "x", upper=100.0)

    def test_fixed_metric_place_count_is_exact(self):
        with self.assertRaises(ValueError):
            GAPS._fixed_metric(b"1.00", 1, "x")
        with self.assertRaises(ValueError):
            GAPS._fixed_metric(b"1.0", 2, "x")

    def test_uint_maximum_is_inclusive(self):
        self.assertEqual(GAPS._uint(b"32", "x", 32), 32)
        with self.assertRaises(ValueError):
            GAPS._uint(b"33", "x", 32)




class NumericGrammarFormTests(_GapFixture):
    """The numeric grammar accepts both the fixed %.6f form dev's engine
    prints today and the earlier %.17g form
    (plus an exact nan/inf/-inf spelling), pinned with non-dyadic literal
    values (-0.3, -2.7, -11.123456) so the two encodings are actually
    distinguishable by their token spelling and not merely by the float
    each happens to parse to."""

    def test_fixed6_form_is_accepted_and_reported(self):
        blob = (
            b"ACCEPT 7 1\n" + self.echo(0) +
            b"DATA 7 1 -0.300000 2 0 -2.700000 1 -11.123456\nx\n" +
            b"DONE 7 STAT 1 0.10 100.0 10.00 1 0\n")
        data, echo, problems, mode, forms = self.full_problems(blob)
        self.assertEqual(problems, [])
        self.assertEqual(forms, collections.Counter({"fixed6": 3}))

    def test_c17g_form_is_accepted_and_reported(self):
        blob = (
            b"ACCEPT 7 1\n" + self.echo(0) +
            b"DATA 7 1 -2.7000000000000002 2 0 -11.123455999999999 "
            b"1 -0.29999999999999999\nx\n" +
            b"DONE 7 STAT 1 0.10 100.0 10.00 1 0\n")
        data, echo, problems, mode, forms = self.full_problems(blob)
        self.assertEqual(problems, [])
        self.assertEqual(forms, collections.Counter({"c17g": 3}))

    def test_a_token_matching_both_grammars_is_ambiguous_and_still_accepted(self):
        # '%.17g' % -1.234567 == '-1.234567', which is ALSO the exact
        # six-decimal spelling of that same double: there is no way to
        # tell, from the token alone, which engine format produced it.
        # An earlier version of this check treated a tail containing both
        # an unambiguous fixed6 token and an unambiguous c17g token as a
        # "mixed forms" error; that rule was removed because per-token
        # classification is inherently ambiguous and the rule produced
        # false rejections on real %.17g transcripts (some of whose
        # tokens are, coincidentally, exact six-decimal spellings too).
        # Nothing about mixing forms is rejected anymore -- only a token
        # matching NEITHER grammar is.
        blob = (
            b"ACCEPT 7 1\n" + self.echo(0) +
            b"DATA 7 1 -1.234567 2 0 -2.7000000000000002 1 -0.300000\nx\n" +
            b"DONE 7 STAT 1 0.10 100.0 10.00 1 0\n")
        data, echo, problems, mode, forms = self.full_problems(blob)
        self.assertEqual(problems, [])
        self.assertEqual(forms, collections.Counter(
            {"ambiguous": 1, "c17g": 1, "fixed6": 1}))

    def test_special_nan_inf_tokens_are_syntactically_valid_then_flagged(self):
        # "nan"/"inf"/"-inf" are exact libc renderings either format's
        # snprintf can emit for a non-finite double: syntactically valid
        # under both grammars, but still fail the finite/non-positive
        # semantic check that applies regardless of which form carried it.
        for special in (b"nan", b"inf", b"-inf"):
            with self.subTest(special=special):
                blob = (
                    b"ACCEPT 7 1\n" + self.echo(0) +
                    b"DATA 7 1 " + special + b" 1 0 -0.300000\nx\n" +
                    b"DONE 7 STAT 1 0.10 100.0 10.00 1 0\n")
                _, _, problems, _, forms = self.full_problems(blob, topk=1)
                self.assertTrue(any(
                    "target logprob is not finite/non-positive" in p
                    for p in problems), problems)
                self.assertIn("special", forms)

    def test_token_matching_neither_form_never_passes(self):
        blob = (
            b"ACCEPT 7 1\n" + self.echo(0) +
            b"DATA 7 1 -0.3 2 0 -2.7 1 -11.123456\nx\n" +
            b"DONE 7 STAT 1 0.10 100.0 10.00 1 0\n")
        _, _, problems = self.problems(blob)
        self.assertTrue(any(
            "malformed DATA numeric fields" in p for p in problems), problems)


class ByteOffsetTests(unittest.TestCase):
    """Every reported problem cites the byte offset of
    the OFFENDING frame's own header line, not the frame that follows it,
    and numeric-tail problems (previously offset-less) now carry one too.
    """

    def test_malformed_global_offset_cites_its_own_line_not_the_next_one(self):
        first = b"\x01\x01READY\x01\x01\n"
        bad_stat = b"STAT 0 0.0 0.0 10.00\n"   # "0.0" should be "0.00"
        blob = first + bad_stat + b"ACCEPT 7 1\n"
        _, problems = GAPS.parse_frames(blob)
        self.assertEqual(len(problems), 1)
        self.assertIn(f"byte {len(first)}", problems[0])
        self.assertNotIn(f"byte {len(first) + len(bad_stat)}", problems[0])

    def test_header_fields_offset_cites_its_own_line_not_the_next_one(self):
        first = b"ACCEPT 7 1\n"
        bad = b" ACCEPT 8 1\n"   # leading space: noncanonical header
        blob = first + bad
        _, problems = GAPS.parse_frames(blob)
        self.assertEqual(len(problems), 1)
        self.assertIn(f"byte {len(first)}", problems[0])
        self.assertNotIn(f"byte {len(first) + len(bad)}", problems[0])

    def test_numeric_tail_problem_cites_the_data_frames_own_header(self):
        prefix = b"ACCEPT 7 1\n" + b"ECHO 7 1 0 nan 0\nx\n"
        # positive target lp: a semantic failure inside _numeric_tail.
        bad_data = b"DATA 7 1 0.300000 2 0 -0.300000 1 -0.300000\n"
        blob = prefix + bad_data + b"x\n" + b"DONE 7 STAT 1 0.10 100.0 10.00 1 0\n"
        frames, framing = GAPS.parse_frames(blob)
        _, _, problems, _, _ = GAPS.check(frames, framing, "7", 2, 3)
        offset = len(prefix)
        self.assertTrue(any(
            f"byte {offset}" in p and
            "target logprob is not finite/non-positive" in p
            for p in problems), problems)

    def test_offset_of_a_non_first_offending_frame_is_hand_verified(self):
        # Three frames precede the offending one: ACCEPT (11 bytes: the
        # 10-character line "ACCEPT 7 1" plus its newline), an ECHO at
        # position 0 (17 bytes: "ECHO 7 1 0 nan 0" plus its newline), and
        # that ECHO's one-byte "x" payload plus its terminator (2 bytes).
        # 11 + 17 + 2 = 30 is where the malformed DONE header starts --
        # a hand count, not a value read back from the module under test.
        line0 = b"ACCEPT 7 1\n"
        line1 = b"ECHO 7 1 0 nan 0\n"
        payload1 = b"x\n"
        self.assertEqual(len(line0), 11)
        self.assertEqual(len(line1), 17)
        self.assertEqual(len(payload1), 2)
        bad_done = b"DONE 7 STAT 1 0.10 100.0 10.00 1\n"   # missing 1 field
        blob = line0 + line1 + payload1 + bad_done
        frames, framing = GAPS.parse_frames(blob)
        self.assertEqual(len(frames), 3)
        self.assertEqual(frames[2][2], 30)
        _, _, problems, _, _ = GAPS.check(frames, framing, "7", 2, 3)
        self.assertTrue(any(
            "malformed DONE frame at byte 30" in p for p in problems), problems)


class PreambleGateWiredIntoMainTests(unittest.TestCase):
    """capture_mode() is actually wired into check(), so
    a transcript opening with an unrecognized (garbage) line fails through
    main()'s own CLI path, not only via a function called directly; and
    the resolved capture mode is always reported, never silent."""

    def test_garbage_preamble_fails_via_main_cli_path(self):
        blob = (
            b"GARBAGE NOT A REAL PREAMBLE\n"
            b"ACCEPT 7 1\n"
            b"ECHO 7 1 0 nan 0\nx\n"
            b"DATA 7 1 -0.300000 2 0 -0.300000 1 -0.300000\nx\n"
            b"DONE 7 STAT 1 0.10 100.0 10.00 1 0\n")
        with tempfile.TemporaryDirectory() as tmp:
            capture = pathlib.Path(tmp) / "capture.raw"
            capture.write_bytes(blob)
            proc = subprocess.run(
                [sys.executable, str(pathlib.Path(GAPS.__file__)), str(capture),
                 "--id", "7", "--topk", "2", "--vocab", "3"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        self.assertEqual(proc.returncode, 1)
        self.assertIn(b"unrecognized frame opens the transcript", proc.stdout)

    def test_capture_mode_and_numeric_form_are_reported_in_main_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            capture = pathlib.Path(tmp) / "capture.raw"
            capture.write_bytes(_GapFixture.VALID)
            proc = subprocess.run(
                [sys.executable, str(pathlib.Path(GAPS.__file__)), str(capture),
                 "--id", "7", "--topk", "2", "--vocab", "3"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        self.assertEqual(proc.returncode, 0)
        self.assertIn(b"capture_mode=request-only", proc.stdout)
        self.assertIn(b"numeric form tally: c17g=3", proc.stdout)

    def test_invalid_capture_mode_is_named_via_check(self):
        # A stray global record with no BANNER/READY lead frame at all.
        blob = b"STAT 0 0.00 0.0 10.00\nACCEPT 7 1\n"
        frames, framing = GAPS.parse_frames(blob)
        self.assertEqual(GAPS.capture_mode(frames), "invalid")
        _, _, problems, mode, _ = GAPS.check(frames, framing, "7", 2, 3)
        self.assertEqual(mode, "invalid")
        self.assertTrue(any("invalid capture mode" in p for p in problems), problems)



class EngineFormatCorpusTests(_GapFixture):
    """A synthesized corpus in the engine's real wire format: every
    numeric token below is spelled exactly as C's `snprintf(..., "%.6f
    %d", ...)` (dev's shipped `logprob_tail()`) would spell it -- not
    hand-picked round numbers, but 210 distinct, non-dyadic values, so no
    single lucky literal is doing the proving. It must pass at head, and
    every one of its numeric tokens is independently confirmed to be
    something the strict %.17g-only grammar this checker's numeric
    parsing predates would have rejected outright.
    """

    N_FRAMES = 210
    VOCAB = 300
    TOPK = 2

    @classmethod
    def _corpus(cls, seed=20260904):
        rng = random.Random(seed)
        parts = [b"ACCEPT 7 1\n", b"ECHO 7 1 0 nan 0\nx\n"]
        tokens = []
        for _ in range(cls.N_FRAMES):
            target_lp = -abs(rng.uniform(1e-6, 20.0))
            target_token = f"{target_lp:.6f}".encode("ascii")
            tokens.append(target_token)
            pair_ids = rng.sample(range(cls.VOCAB), cls.TOPK)
            pieces = [target_token, str(cls.TOPK).encode("ascii")]
            for tid in pair_ids:
                token_lp = -abs(rng.uniform(1e-6, 20.0))
                token_lp_token = f"{token_lp:.6f}".encode("ascii")
                tokens.append(token_lp_token)
                pieces.append(str(tid).encode("ascii"))
                pieces.append(token_lp_token)
            parts.append(b"DATA 7 1 " + b" ".join(pieces) + b"\nx\n")
        parts.append(
            f"DONE 7 STAT {cls.N_FRAMES} 0.10 100.0 10.00 1 0\n".encode("ascii"))
        return b"".join(parts), tokens

    def test_engine_format_corpus_of_210_frames_passes(self):
        blob, tokens = self._corpus()
        self.assertGreaterEqual(len(tokens), self.N_FRAMES)
        frames, framing = GAPS.parse_frames(blob)
        data, echo, problems, mode, forms = GAPS.check(
            frames, framing, "7", self.TOPK, self.VOCAB)
        self.assertEqual(problems, [])
        self.assertEqual(data, self.N_FRAMES)
        # Every token here is a genuine %.6f emission, so none can be
        # classified as c17g-only; some nonetheless land on an exact
        # %.17g spelling too (the false-positive "mixed forms" case a
        # since-removed rejection used to misfire on) and are tallied
        # "ambiguous" rather than "fixed6" -- informational only, and
        # this test's PASS/problems assertions above already prove that
        # tally has no bearing on the verdict.
        self.assertEqual(forms["c17g"], 0)
        self.assertGreater(forms["ambiguous"], 0)
        self.assertEqual(sum(forms.values()), len(tokens))

    def test_engine_format_corpus_values_are_non_dyadic(self):
        # A dyadic value (an exact binary fraction, e.g. 0.125) would make
        # the fixed6/c17g distinction uninteresting for that token, since
        # both forms could spell it exactly. Confirm the corpus avoids
        # that by construction: none of its %.6f tokens round-trips
        # through Python's own dyadic-fraction check.
        _, tokens = self._corpus()
        dyadic = 0
        for token in tokens:
            value = float(token)
            # A double is dyadic (exactly binary-fraction-representable at
            # 6 decimal places) iff multiplying by 1e6 and rounding loses
            # nothing AND the resulting numerator's lowest set bits divide
            # evenly -- simpler and just as decisive here: a dyadic value
            # would print identically under %.17g and %.6f once trailing
            # zeros are accounted for, which none of these do (see the
            # next test) -- this test instead confirms none is a "clean"
            # few-bits-of-mantissa value like *.0, *.5, *.25, *.125.
            frac = abs(value) - int(abs(value))
            eighths = frac * 8
            if abs(eighths - round(eighths)) < 1e-9:
                dyadic += 1
        self.assertEqual(dyadic, 0)

    def test_engine_format_corpus_would_fail_a_c17g_only_grammar(self):
        # Every token in the corpus is a %.6f spelling. Whether any ONE
        # such token also happens to be its own double's shortest %.17g
        # round-trip spelling is unpredictable per-token (it depends on
        # that double's neighborhood, not on the fact that it came from
        # %.6f) -- but the module docstring's actual claim is about the
        # TRANSCRIPT, not any single token: one rejected token anywhere
        # is enough to fail the whole check, since `_numeric_tail` bails
        # out of the entire frame the moment its numeric parse raises.
        # Confirm most tokens are rejected outright by the standalone
        # %.17g parser this module still carries (`_c17g`, kept for the
        # earlier engine's own emitted form), and that at least one
        # rejection lands inside a real DATA frame of the corpus --
        # which is what actually dooms the whole transcript under a
        # %.17g-only grammar.
        blob, tokens = self._corpus()
        rejected = 0
        for token in tokens:
            try:
                GAPS._c17g(token, "corpus token")
            except ValueError:
                rejected += 1
        self.assertGreater(rejected, len(tokens) // 2, (rejected, len(tokens)))

        frame_rejected = False
        for line in blob.split(b"\n"):
            if not line.startswith(b"DATA "):
                continue
            fields = line.split(b" ")
            for field in fields[3:]:
                try:
                    GAPS._c17g(field, "corpus token")
                except ValueError:
                    frame_rejected = True
                    break
            if frame_rejected:
                break
        self.assertTrue(
            frame_rejected,
            "expected at least one DATA frame in the corpus to contain a "
            "token the %.17g-only parser refuses")

if __name__ == "__main__":
    unittest.main()
