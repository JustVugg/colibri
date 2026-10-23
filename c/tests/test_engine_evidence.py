"""tools/engine_evidence.py must parse only exact, in-range preamble text.

Pins the exact banner/loaded text the module accepts, the field ranges it
enforces on both sides of each bound, and the None-vs-raise split in
parse_engine_preamble, so a future edit to the shared parser cannot
silently loosen or break any of its numeric bounds or its exact-text
matching without a local, fast failure.
"""
import unittest

from tools.engine_evidence import (
    IDOT_KERNELS,
    PreambleError,
    parse_engine_banner,
    parse_engine_loaded,
    parse_engine_preamble,
)

_BANNER = (
    "== GLM C engine (glm_moe_dsa), cache=8 experts/layer | "
    "compute experts@4-bit dense@8-bit | idot: avx2 =="
)
_LOADED = (
    "loaded in 12.34s | resident dense: 5678.90 MB | layers=32 experts=128 "
    "| MTP ACTIVE (draft=4)"
)


def _banner(**subs):
    text = _BANNER
    for old, new in subs.items():
        assert old in text, old
        text = text.replace(old, new, 1)
    return text


def _loaded(**subs):
    text = _LOADED
    for old, new in subs.items():
        assert old in text, old
        text = text.replace(old, new, 1)
    return text


class ParseEngineBannerTest(unittest.TestCase):
    def test_exact_banner_returns_typed_fields(self):
        fields = parse_engine_banner(_BANNER)
        self.assertEqual(fields, {
            "kind": "BANNER", "cap": 8, "expert_bits": 4, "dense_bits": 8,
            "kernel": "avx2",
        })

    def test_non_string_raises(self):
        with self.assertRaises(PreambleError):
            parse_engine_banner(None)

    def test_unrecognized_text_raises(self):
        with self.assertRaises(PreambleError):
            parse_engine_banner("not a banner at all")

    def test_unknown_kernel_raises(self):
        # Negative control for the roster test below: an unlisted kernel
        # name (real ISA extension, not in IDOT_KERNELS) must be refused.
        with self.assertRaises(PreambleError):
            parse_engine_banner(_banner(**{"idot: avx2": "idot: sse4"}))

    def test_trailing_text_rejected(self):
        with self.assertRaises(PreambleError):
            parse_engine_banner(_BANNER + " extra")

    def test_trailing_newline_rejected(self):
        # fullmatch requires consuming the WHOLE string; $ is zero-width
        # and matches just before a trailing "\n" too, so a weakening to
        # .match() or .search() would let a newline-terminated banner
        # through here even though fullmatch correctly refuses it. A
        # caller doing `for line in f:` on real engine stdout hands over
        # lines WITH their trailing newline, so this is the shape a real
        # caller would actually feed in, not a synthetic corner case.
        with self.assertRaises(PreambleError):
            parse_engine_banner(_BANNER + "\n")

    # -- cap: [1, 2**31-1] --

    def test_cap_lower_bound_accepted(self):
        fields = parse_engine_banner(_banner(**{"cache=8": "cache=1"}))
        self.assertEqual(fields["cap"], 1)

    def test_cap_lower_bound_rejected(self):
        with self.assertRaises(PreambleError):
            parse_engine_banner(_banner(**{"cache=8": "cache=0"}))

    def test_cap_upper_bound_accepted(self):
        fields = parse_engine_banner(
            _banner(**{"cache=8": "cache=2147483647"}))
        self.assertEqual(fields["cap"], 2147483647)

    def test_cap_upper_bound_rejected(self):
        with self.assertRaises(PreambleError):
            parse_engine_banner(_banner(**{"cache=8": "cache=2147483648"}))

    # -- expert_bits: [1, 16] --

    def test_expert_bits_lower_bound_accepted(self):
        fields = parse_engine_banner(_banner(**{"experts@4-bit": "experts@1-bit"}))
        self.assertEqual(fields["expert_bits"], 1)

    def test_expert_bits_lower_bound_rejected(self):
        with self.assertRaises(PreambleError):
            parse_engine_banner(_banner(**{"experts@4-bit": "experts@0-bit"}))

    def test_expert_bits_upper_bound_accepted(self):
        fields = parse_engine_banner(_banner(**{"experts@4-bit": "experts@16-bit"}))
        self.assertEqual(fields["expert_bits"], 16)

    def test_expert_bits_upper_bound_rejected(self):
        with self.assertRaises(PreambleError):
            parse_engine_banner(_banner(**{"experts@4-bit": "experts@17-bit"}))

    # -- dense_bits: [1, 16] --

    def test_dense_bits_lower_bound_accepted(self):
        fields = parse_engine_banner(_banner(**{"dense@8-bit": "dense@1-bit"}))
        self.assertEqual(fields["dense_bits"], 1)

    def test_dense_bits_lower_bound_rejected(self):
        with self.assertRaises(PreambleError):
            parse_engine_banner(_banner(**{"dense@8-bit": "dense@0-bit"}))

    def test_dense_bits_upper_bound_accepted(self):
        fields = parse_engine_banner(_banner(**{"dense@8-bit": "dense@16-bit"}))
        self.assertEqual(fields["dense_bits"], 16)

    def test_dense_bits_upper_bound_rejected(self):
        with self.assertRaises(PreambleError):
            parse_engine_banner(_banner(**{"dense@8-bit": "dense@17-bit"}))

    # -- leading-zero handling (no field allows a leading zero on a
    #    multi-digit value; a leading zero makes the whole line unrecognized,
    #    not merely out of range) --

    def test_leading_zero_digit_rejected(self):
        with self.assertRaises(PreambleError):
            parse_engine_banner(_banner(**{"cache=8": "cache=007"}))

    def test_no_leading_zero_digit_accepted(self):
        fields = parse_engine_banner(_banner(**{"cache=8": "cache=7"}))
        self.assertEqual(fields["cap"], 7)

    def test_oversized_field_raises_preamble_error_not_bare_value_error(self):
        # _UINT_TEXT has no digit-count cap, so a field beyond Python's
        # int-string conversion limit (sys.int_info.default_max_str_digits,
        # 4300 by default) used to reach int()/float() raw, escaping as a
        # bare ValueError instead of this module's own PreambleError --
        # breaking every caller's contract to refuse with a named error.
        huge = "1" + "0" * 4300
        with self.assertRaises(PreambleError):
            parse_engine_banner(_banner(**{"cache=8": f"cache={huge}"}))


class IdotKernelRosterTest(unittest.TestCase):
    """Pin every IDOT_KERNELS entry from the engine's own source, not just
    avx2 (the only entry any test previously referenced -- confirmed by
    RAN mutation: reducing IDOT_KERNELS to ("avx2",) alone left this
    module's OWN 45-test suite fully green, since nothing here iterated
    or symbolically referenced the roster). Removing any single entry
    below, or the roster itself down to fewer entries, must fail this
    test -- proof-of-bite for each is in the worker report.

    Source of the 7 entries, cited by file:line: c/quant.h:582-594's
    #if/#elif ladder defines IDOT_KERNEL to each of these string
    literals in turn, and c/colibri.c:11243 prints it unmodified inside
    the exact banner format this fixture reproduces character-for-
    character (cache=%d experts/layer | compute experts@%d-bit
    dense@%d-bit | idot: <IDOT_KERNEL>).

      c/quant.h:582  "avx512-vnni"  (__AVX512VNNI__ && __AVX512BW__)
      c/quant.h:584  "avx-vnni"     (__AVXVNNI__ && __AVX2__)
      c/quant.h:586  "avx2"         (__AVX2__)
      c/quant.h:588  "neon-i8mm"    (__ARM_NEON && __ARM_FEATURE_MATMUL_INT8)
      c/quant.h:590  "neon"         (__ARM_NEON, no i8mm)               -- RAN
      c/quant.h:592  "vsx"          (__VSX__)
      c/quant.h:594  "scalar"       (else)

    "neon" is marked RAN: this development host is arm64 Darwin, and
    `echo | cc -dM -E -` (no -mcpu, matching this project's own Makefile
    comment "ARCH unset -> no -mcpu, default build byte-identical")
    defines __ARM_NEON but NOT __ARM_FEATURE_MATMUL_INT8 on this
    machine, so its own default (unaccelerated) build takes the "neon"
    branch of the ladder above -- confirmed by running that exact
    compiler invocation, not by reading the source alone. This is a
    compiler-preprocessor probe of the flags the default build actually
    uses, not a captured engine stdout banner: no engine binary was
    built or run for this (no model runs, per the dispatch's hard
    bound). The other six entries are taken from the C source only
    (INFERRED from c/quant.h's literals, not independently reproduced on
    real hardware for each ISA).

    VACUOUS-GATE NOTE: test_every_roster_entry_parses_from_real_banner_text
    below iterates IDOT_KERNELS and checks each entry parses to itself --
    but _BANNER_RE is ITSELF built from IDOT_KERNELS
    ("|".join(IDOT_KERNELS)), so a renamed entry (same count, different
    spelling) produces a banner the correspondingly-mutated regex still
    matches: the assertion compares the mutation against itself and can
    pass having checked nothing about the entry's real spelling. Only
    test_roster_matches_frozen_expectation below, which compares against
    an independent literal transcribed from c/quant.h rather than against
    IDOT_KERNELS itself, actually defends the CONTENT of each entry; the
    per-entry parse loop is kept because it still defends something real
    (that parse_engine_banner's kernel-dispatch group and IDOT_KERNELS
    stay in sync with each other), just not entry spelling on its own.
    """

    def test_every_roster_entry_parses_from_real_banner_text(self):
        for kernel in IDOT_KERNELS:
            with self.subTest(kernel=kernel):
                fields = parse_engine_banner(
                    _banner(**{"idot: avx2": f"idot: {kernel}"}))
                self.assertEqual(fields["kernel"], kernel)

    def test_roster_matches_frozen_expectation(self):
        # Frozen from c/quant.h:582-594's #if/#elif ladder. If the
        # engine's ladder changes, this literal and the roster both
        # change, deliberately and together. Unlike the per-entry parse
        # loop above, this does NOT derive its expectation from
        # IDOT_KERNELS or from _BANNER_RE (which is itself built from
        # IDOT_KERNELS) -- it is the independent source transcription
        # that makes a same-length rename of any entry (which the parse
        # loop and the count/uniqueness check below both miss) fail.
        EXPECTED_IDOT_KERNELS = ("avx512-vnni", "avx-vnni", "avx2",
                                 "neon-i8mm", "neon", "vsx", "scalar")
        self.assertEqual(IDOT_KERNELS, EXPECTED_IDOT_KERNELS)

    def test_roster_is_not_accidentally_empty_or_singleton(self):
        # A cheap sanity backstop for the roster itself, independent of
        # any one entry's own test above. Redundant with the frozen-
        # literal test for a length change, but kept because it is a
        # different, cheaper check that would survive even if the
        # literal above ever needed updating for a real ladder change.
        self.assertEqual(len(IDOT_KERNELS), 7)
        self.assertEqual(len(set(IDOT_KERNELS)), 7)


class ParseEngineLoadedTest(unittest.TestCase):
    def test_exact_loaded_returns_typed_fields(self):
        fields = parse_engine_loaded(_LOADED)
        self.assertEqual(fields, {
            "kind": "LOADED", "load_s": 12.34, "resident_mb": 5678.90,
            "layers": 32, "experts": 128, "mtp": "ACTIVE", "draft": 4,
        })

    def test_non_string_raises(self):
        with self.assertRaises(PreambleError):
            parse_engine_loaded(1234)

    def test_unrecognized_text_raises(self):
        with self.assertRaises(PreambleError):
            parse_engine_loaded("not a load record")

    def test_trailing_newline_rejected(self):
        # See ParseEngineBannerTest.test_trailing_newline_rejected: same
        # fullmatch-vs-$ subtlety, same real-caller shape (stdout lines
        # iterated with their newline still attached).
        with self.assertRaises(PreambleError):
            parse_engine_loaded(_LOADED + "\n")

    # -- layers: [1, 128] --

    def test_layers_lower_bound_accepted(self):
        fields = parse_engine_loaded(_loaded(**{"layers=32": "layers=1"}))
        self.assertEqual(fields["layers"], 1)

    def test_layers_lower_bound_rejected(self):
        with self.assertRaises(PreambleError):
            parse_engine_loaded(_loaded(**{"layers=32": "layers=0"}))

    def test_layers_upper_bound_accepted(self):
        fields = parse_engine_loaded(_loaded(**{"layers=32": "layers=128"}))
        self.assertEqual(fields["layers"], 128)

    def test_layers_upper_bound_rejected(self):
        with self.assertRaises(PreambleError):
            parse_engine_loaded(_loaded(**{"layers=32": "layers=129"}))

    # -- experts: [1, 4096] --

    def test_experts_lower_bound_accepted(self):
        fields = parse_engine_loaded(_loaded(**{"experts=128": "experts=1"}))
        self.assertEqual(fields["experts"], 1)

    def test_experts_lower_bound_rejected(self):
        with self.assertRaises(PreambleError):
            parse_engine_loaded(_loaded(**{"experts=128": "experts=0"}))

    def test_experts_upper_bound_accepted(self):
        fields = parse_engine_loaded(_loaded(**{"experts=128": "experts=4096"}))
        self.assertEqual(fields["experts"], 4096)

    def test_experts_upper_bound_rejected(self):
        with self.assertRaises(PreambleError):
            parse_engine_loaded(_loaded(**{"experts=128": "experts=4097"}))

    # -- exactly two decimal digits on load_s / resident_mb --

    def test_two_decimal_places_accepted(self):
        fields = parse_engine_loaded(_LOADED)
        self.assertEqual(fields["load_s"], 12.34)

    def test_one_decimal_place_rejected(self):
        with self.assertRaises(PreambleError):
            parse_engine_loaded(_loaded(**{"12.34s": "12.3s"}))

    def test_three_decimal_places_rejected(self):
        with self.assertRaises(PreambleError):
            parse_engine_loaded(_loaded(**{"12.34s": "12.345s"}))

    # -- MTP / draft interaction --

    def test_absent_mtp_allows_nonzero_draft(self):
        fields = parse_engine_loaded(
            _loaded(**{"MTP ACTIVE (draft=4)": "MTP absent (draft=5)"}))
        self.assertEqual(fields["mtp"], "absent")
        self.assertEqual(fields["draft"], 5)

    def test_active_mtp_allows_nonzero_draft(self):
        fields = parse_engine_loaded(_LOADED)
        self.assertEqual(fields["mtp"], "ACTIVE")
        self.assertEqual(fields["draft"], 4)

    def test_draft_upper_bound_accepted(self):
        fields = parse_engine_loaded(_loaded(**{"draft=4)": "draft=63)"}))
        self.assertEqual(fields["draft"], 63)

    def test_draft_upper_bound_rejected(self):
        with self.assertRaises(PreambleError):
            parse_engine_loaded(_loaded(**{"draft=4)": "draft=64)"}))

    def test_disabled_multiplexed_requires_zero_draft(self):
        with self.assertRaises(PreambleError):
            parse_engine_loaded(_loaded(
                **{"MTP ACTIVE (draft=4)":
                   "MTP DISABLED (multiplexed serve) (draft=4)"}))

    def test_disabled_multiplexed_with_zero_draft_parses(self):
        fields = parse_engine_loaded(_loaded(
            **{"MTP ACTIVE (draft=4)":
               "MTP DISABLED (multiplexed serve) (draft=0)"}))
        self.assertEqual(fields["mtp"], "DISABLED (multiplexed serve)")
        self.assertEqual(fields["draft"], 0)

    def test_oversized_field_raises_preamble_error_not_bare_value_error(self):
        # See ParseEngineBannerTest's identical test: same digit-count
        # cap gap, same fix, this function's own layers field.
        huge = "1" + "0" * 4300
        with self.assertRaises(PreambleError):
            parse_engine_loaded(_loaded(**{"layers=32": f"layers={huge}"}))


class ParseEnginePreambleTest(unittest.TestCase):
    def test_dispatches_to_banner(self):
        self.assertEqual(
            parse_engine_preamble(_BANNER), parse_engine_banner(_BANNER))

    def test_dispatches_to_loaded(self):
        self.assertEqual(
            parse_engine_preamble(_LOADED), parse_engine_loaded(_LOADED))

    def test_unowned_line_returns_none(self):
        self.assertIsNone(parse_engine_preamble("some ordinary log line"))

    def test_banner_prefixed_but_malformed_still_raises(self):
        with self.assertRaises(PreambleError):
            parse_engine_preamble("== GLM C engine but garbled ==")

    def test_loaded_prefixed_but_malformed_still_raises(self):
        with self.assertRaises(PreambleError):
            parse_engine_preamble("loaded in not a valid record")

    def test_non_string_raises(self):
        with self.assertRaises(PreambleError):
            parse_engine_preamble(3.14)


if __name__ == "__main__":
    unittest.main()
