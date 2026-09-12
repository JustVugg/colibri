"""tools/engine_evidence.py must parse only exact, in-range preamble text.

Pins the exact banner/loaded text the module accepts, the field ranges it
enforces on both sides of each bound, and the None-vs-raise split in
parse_engine_preamble, so a future edit to the shared parser cannot
silently loosen or break any of its numeric bounds or its exact-text
matching without a local, fast failure.
"""
import unittest

from tools.engine_evidence import (
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
        with self.assertRaises(PreambleError):
            parse_engine_banner(_banner(**{"idot: avx2": "idot: sse4"}))

    def test_trailing_text_rejected(self):
        with self.assertRaises(PreambleError):
            parse_engine_banner(_BANNER + " extra")

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
