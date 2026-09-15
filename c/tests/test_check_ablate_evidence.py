"""tools/check_ablate_evidence.py must accept only a complete, self-binding
ABLATE evidence artifact and reject every other input: a truncated or
replayed record stream, a header that does not bind the manifest or the
external config.json it is checked against, any field with the wrong
type, range, or key set at any of the four record kinds (header, item
header, target row, terminal completion), and a target row whose fields
contradict one another in a way the producer can never emit.

Checks enumerated from the source (`tools/check_ablate_evidence.py`,
read in full before writing this module) and covered below, grouped by
the function that performs them:

- `_checked_engine_text_size` / `_bounded_config_bytes`: the 256 MiB
  inclusive engine text limit, both sides.
- `_reject_constant`, `_reject_duplicate_keys`, `_json_record`: no
  NaN/Infinity JSON constants, no duplicate object keys, invalid
  JSON/non-ASCII text rejected.
- `_config_identity`: empty file; invalid JSON; non-object root; each
  of vocab_size/num_hidden_layers/n_routed_experts/first_k_dense_replace
  missing or out of range.
- `_manifest_proof`: framing -- an empty manifest, an empty record, a
  carriage return inside a record and an embedded NUL are refused, while
  CRLF endings and a missing final newline are accepted and reduced to
  the canonical form the engine binds; non-ASCII line; non-canonical integer grammar; too few fields;
  every per-item field bound (item id, T, prompt, mode, cell count);
  the mode/cell-count pairing rule; the field-count/denominator
  arithmetic; every per-cell bound (layer, expert, applied-target,
  mode-3 vs other-mode applied-target rule, duplicate cell);
  out-of-vocabulary tokens; duplicate item ids across lines.
- `validate_ablate_evidence`: the evidence framing check; the header's
  key set, type, and range checks; the header-vs-config identity
  check; the header-vs-manifest-proof binding check; the item header's
  key set, type, and manifest-order check; the target row's key set,
  type, and identity checks; the three cross-field invariants (below);
  the top-k list's shape, range, and
  uniqueness checks; the terminal record's key set, bounds, and exact
  content check; missing/extra/trailing records at every boundary.

Every fixture here is a literal artifact built by hand from the
module's documented wire schema (`coli-ablate/2`) and hashed with the
stdlib `hashlib` directly -- no expected value is produced by calling
the validator under test.
"""
import copy
import hashlib
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from tools import check_ablate_evidence as ABLATE
from tools import engine_evidence

_DOMAIN = b"coli-ablate-manifest/2\n"


def _serialize(records):
    return b"".join(
        json.dumps(record, separators=(",", ":")).encode("ascii") + b"\n"
        for record in records)


def _write(root, manifest_raw, evidence_raw, config_raw):
    manifest = root / "manifest.txt"
    evidence = root / "evidence.jsonl"
    config = root / "config.json"
    manifest.write_bytes(manifest_raw)
    evidence.write_bytes(evidence_raw)
    config.write_bytes(config_raw)
    return manifest, evidence, config


def _run_cli(manifest, evidence, config):
    return subprocess.run(
        [sys.executable, str(pathlib.Path(ABLATE.__file__)),
         str(manifest), str(evidence), "--config", str(config)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)


class _GoldenFixture(unittest.TestCase):
    """Shared two-item artifact, hand-derived from the documented schema.

    Manifest: item 1 (T=3, prompt=1, baseline, tokens 0,1,2) then item 2
    (T=2, prompt=1, mode 1 with one ablated cell at layer 1/expert 2,
    tokens 3,0). vocab=4, n_layers=4, first_dense=1, n_experts=5, so
    topk = min(32, 4) = 4. Positions/gold are derived by hand from the
    manifest's own tokens: item 1 has positions [0, 1] with gold tokens
    1 and 2; item 2 has position [0] with gold token 0.
    """

    MANIFEST = b"1 3 1 0 0 0 1 2\n2 2 1 1 1 1 2 -1 3 0\n"
    CONFIG = (b'{"vocab_size":4,"num_hidden_layers":4,'
              b'"first_k_dense_replace":1,"n_routed_experts":5}\n')
    MANIFEST_SHA256 = hashlib.sha256(_DOMAIN + MANIFEST).hexdigest()
    CONFIG_SHA256 = hashlib.sha256(CONFIG).hexdigest()

    @classmethod
    def golden_records(cls):
        header = {
            "t": "hdr", "schema": "coli-ablate/2", "vocab": 4, "topk": 4,
            "n_layers": 4, "first_dense": 1, "n_experts": 5,
            "config_sha256": cls.CONFIG_SHA256,
            "manifest_sha256": cls.MANIFEST_SHA256,
            "expected_items": 2, "expected_targets": 3,
        }
        item1_header = {"t": "ah", "item": 1, "mode": 0, "ncells": 0,
                        "T": 3, "n_prompt": 1, "cells": []}
        row1 = {"t": "lg", "item": 1, "pos": 0, "gold": 1,
                "nll": 0.2, "glogit": 1.0, "molo": 0.5, "mgn": 0.5,
                "am": 1, "amlogit": 1.0, "logZ": 1.3, "corr": 1,
                "tk": [[0, 0.1], [1, 1.0], [2, 0.3], [3, -0.2]]}
        row2 = {"t": "lg", "item": 1, "pos": 1, "gold": 2,
                "nll": 0.7, "glogit": 0.4, "molo": 0.9, "mgn": -0.5,
                "am": 0, "amlogit": 0.9, "logZ": 1.1, "corr": 0,
                "tk": [[0, 0.9], [1, 0.1], [2, 0.4], [3, -0.3]]}
        item2_header = {"t": "ah", "item": 2, "mode": 1, "ncells": 1,
                        "T": 2, "n_prompt": 1, "cells": [[1, 2, -1]]}
        row3 = {"t": "lg", "item": 2, "pos": 0, "gold": 0,
                "nll": 0.0, "glogit": 2.0, "molo": -1e30, "mgn": 1e30,
                "am": 0, "amlogit": 2.0, "logZ": 2.0, "corr": 1,
                "tk": [[0, 2.0], [1, -1.0], [2, -2.0], [3, -3.0]]}
        done = {"t": "done", "manifest_sha256": cls.MANIFEST_SHA256,
                "completed_items": 2, "completed_targets": 3}
        return [header, item1_header, row1, row2, item2_header, row3, done]

    def _reject(self, mutate, records=None):
        """Apply `mutate` to a deep copy of the golden records and assert
        the mutated artifact is refused."""
        mutated = copy.deepcopy(records if records is not None
                                 else self.golden_records())
        mutate(mutated)
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            manifest, evidence, config = _write(
                root, self.MANIFEST, _serialize(mutated), self.CONFIG)
            with self.assertRaises(ABLATE.AblateEvidenceError):
                ABLATE.validate_ablate_evidence(manifest, evidence, config)


class GoldenArtifactAcceptedTests(_GoldenFixture):
    def test_valid_artifact_is_accepted_and_pass_line_is_exact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            manifest, evidence, config = _write(
                root, self.MANIFEST, _serialize(self.golden_records()),
                self.CONFIG)
            result = ABLATE.validate_ablate_evidence(
                manifest, evidence, config)
            self.assertEqual(result, {
                "manifest_sha256": self.MANIFEST_SHA256,
                "items": 2, "targets": 3,
            })
            cli = _run_cli(manifest, evidence, config)
            self.assertEqual(cli.returncode, 0, cli.stderr.decode(errors="replace"))
            self.assertEqual(cli.stderr, b"")
            # print() terminates the line with the platform's newline, so a
            # Windows child hands back CRLF; the pin is on the line's content.
            self.assertEqual(
                cli.stdout.replace(b"\r\n", b"\n"),
                f"[ablate-evidence] PASS manifest={self.MANIFEST_SHA256} "
                f"items=2 targets=3\n".encode("ascii"))


class TopkProducerCapAboveVocabFourTests(unittest.TestCase):
    """The header `topk == min(32, vocab)` check only ever exercises the
    "vocab is the binding constraint" side at `_GoldenFixture`'s vocab=4.
    This fixture uses vocab=40 (above both 4 and the 32 cap) to pin the
    other side: topk must be capped at 32, not left equal to vocab.
    """

    MANIFEST = b"1 2 1 0 0 0 39\n"
    CONFIG = (b'{"vocab_size":40,"num_hidden_layers":1,'
              b'"first_k_dense_replace":0,"n_routed_experts":1}\n')
    MANIFEST_SHA256 = hashlib.sha256(_DOMAIN + MANIFEST).hexdigest()
    CONFIG_SHA256 = hashlib.sha256(CONFIG).hexdigest()

    @classmethod
    def golden_records(cls, topk=32, tk_count=32):
        header = {
            "t": "hdr", "schema": "coli-ablate/2", "vocab": 40, "topk": topk,
            "n_layers": 1, "first_dense": 0, "n_experts": 1,
            "config_sha256": cls.CONFIG_SHA256,
            "manifest_sha256": cls.MANIFEST_SHA256,
            "expected_items": 1, "expected_targets": 1,
        }
        item1_header = {"t": "ah", "item": 1, "mode": 0, "ncells": 0,
                        "T": 2, "n_prompt": 1, "cells": []}
        row = {"t": "lg", "item": 1, "pos": 0, "gold": 39,
               "nll": 0.0, "glogit": 1.0, "molo": 0.5, "mgn": 0.5,
               "am": 39, "amlogit": 1.0, "logZ": 1.3, "corr": 1,
               "tk": [[i, -0.01 * i] for i in range(tk_count)]}
        done = {"t": "done", "manifest_sha256": cls.MANIFEST_SHA256,
                "completed_items": 1, "completed_targets": 1}
        return [header, item1_header, row, done]

    def test_topk_capped_at_32_for_vocab_above_the_cap_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            manifest, evidence, config = _write(
                root, self.MANIFEST, _serialize(self.golden_records()),
                self.CONFIG)
            result = ABLATE.validate_ablate_evidence(
                manifest, evidence, config)
            self.assertEqual(result, {
                "manifest_sha256": self.MANIFEST_SHA256,
                "items": 1, "targets": 1,
            })

    def test_topk_left_uncapped_at_vocab_above_32_is_rejected(self):
        # vocab=40 > 32, so header topk must be 32 -- not 40 (== vocab).
        records = self.golden_records(topk=40, tk_count=40)
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            manifest, evidence, config = _write(
                root, self.MANIFEST, _serialize(records), self.CONFIG)
            with self.assertRaises(ABLATE.AblateEvidenceError):
                ABLATE.validate_ablate_evidence(manifest, evidence, config)


class EvidenceFramingTests(_GoldenFixture):
    """`validate_ablate_evidence`'s canonical-LF-JSONL framing check."""

    def _reject_raw(self, evidence_raw):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            manifest, evidence, config = _write(
                root, self.MANIFEST, evidence_raw, self.CONFIG)
            with self.assertRaises(ABLATE.AblateEvidenceError):
                ABLATE.validate_ablate_evidence(manifest, evidence, config)

    def test_empty_evidence_rejected(self):
        self._reject_raw(b"")

    def test_evidence_missing_trailing_newline_rejected(self):
        self._reject_raw(_serialize(self.golden_records())[:-1])

    def test_evidence_with_carriage_return_rejected(self):
        self._reject_raw(_serialize(self.golden_records()).replace(
            b"\n", b"\r\n", 1))

    def test_evidence_with_nul_byte_rejected(self):
        self._reject_raw(_serialize(self.golden_records()) + b"\0")


class HeaderRecordTests(_GoldenFixture):
    CASES = (
        ("missing_key", lambda r: r[0].pop("topk")),
        ("extra_key", lambda r: r[0].__setitem__("extra", 1)),
        ("wrong_t", lambda r: r[0].__setitem__("t", "nope")),
        ("wrong_schema", lambda r: r[0].__setitem__(
            "schema", "coli-ablate/1")),
        ("topk_wrong_type", lambda r: r[0].__setitem__("topk", "4")),
        ("config_sha256_wrong_type", lambda r: r[0].__setitem__(
            "config_sha256", 1)),
        ("config_sha256_not_hex", lambda r: r[0].__setitem__(
            "config_sha256", "z" * 64)),
        ("manifest_sha256_wrong_type", lambda r: r[0].__setitem__(
            "manifest_sha256", None)),
        ("manifest_sha256_not_hex", lambda r: r[0].__setitem__(
            "manifest_sha256", "0" * 63 + "g")),
        ("vocab_out_of_range", lambda r: r[0].__setitem__("vocab", 0)),
        ("n_layers_out_of_range", lambda r: r[0].__setitem__(
            "n_layers", 0)),
        ("first_dense_out_of_range", lambda r: r[0].__setitem__(
            "first_dense", 99)),
        ("n_experts_out_of_range", lambda r: r[0].__setitem__(
            "n_experts", 0)),
        ("expected_items_out_of_range", lambda r: r[0].__setitem__(
            "expected_items", 0)),
        ("expected_targets_out_of_range", lambda r: r[0].__setitem__(
            "expected_targets", 0)),
        ("topk_not_producer_exact", lambda r: r[0].__setitem__("topk", 3)),
        ("vocab_identity_mismatch", lambda r: (
            r[0].__setitem__("vocab", 5), r[0].__setitem__("topk", 5))),
        ("n_layers_identity_mismatch", lambda r: r[0].__setitem__(
            "n_layers", 2)),
        ("first_dense_identity_mismatch", lambda r: r[0].__setitem__(
            "first_dense", 0)),
        ("n_experts_identity_mismatch", lambda r: r[0].__setitem__(
            "n_experts", 6)),
        ("config_sha256_identity_mismatch", lambda r: r[0].__setitem__(
            "config_sha256", "0" * 64)),
        ("manifest_sha256_binding_mismatch", lambda r: r[0].__setitem__(
            "manifest_sha256", "1" * 64)),
        ("expected_items_binding_mismatch", lambda r: r[0].__setitem__(
            "expected_items", 99)),
        ("expected_targets_binding_mismatch", lambda r: r[0].__setitem__(
            "expected_targets", 99)),
    )

    def test_header_field_checks(self):
        for name, mutate in self.CASES:
            with self.subTest(name=name):
                self._reject(mutate)

    def test_topk_not_producer_exact_even_when_every_row_agrees_with_it(self):
        # Isolates the header-level topk==min(32,vocab) check from the
        # per-row "len(tk) == header['topk']" shape check: here every row's
        # tk list is ALSO shrunk to match the wrong topk, so only the
        # header-level producer-exactness check can catch the artifact.
        def mutate(records):
            records[0]["topk"] = 2
            for record in records:
                if record.get("t") == "lg":
                    record["tk"] = record["tk"][:2]
        self._reject(mutate)


class ItemHeaderRecordTests(_GoldenFixture):
    def test_missing_item_header_rejected(self):
        self._reject(lambda r: r.__delitem__(slice(1, None)))

    def test_item_header_not_a_dict_rejected(self):
        self._reject(lambda r: r.__setitem__(1, 5))

    def test_item_header_missing_key_rejected(self):
        self._reject(lambda r: r[1].pop("ncells"))

    def test_item_header_extra_key_rejected(self):
        self._reject(lambda r: r[1].__setitem__("extra", 1))

    def test_item_header_field_wrong_type_rejected(self):
        self._reject(lambda r: r[1].__setitem__("item", "1"))

    def test_item_header_cells_not_a_list_rejected(self):
        self._reject(lambda r: r[4].__setitem__("cells", {}))

    def test_item_header_cell_wrong_shape_rejected(self):
        self._reject(lambda r: r[4].__setitem__("cells", [[1, 2]]))

    def test_item_header_cell_element_wrong_type_rejected(self):
        self._reject(lambda r: r[4].__setitem__(
            "cells", [[1, 2, "x"]]))

    def test_item_header_mismatch_vs_manifest_rejected(self):
        self._reject(lambda r: r[1].__setitem__("T", 99))


class TargetRowRecordTests(_GoldenFixture):
    def test_missing_target_row_rejected(self):
        self._reject(lambda r: r.__delitem__(slice(2, None)))

    def test_row_not_a_dict_rejected(self):
        self._reject(lambda r: r.__setitem__(2, 5))

    def test_row_missing_key_rejected(self):
        self._reject(lambda r: r[2].pop("corr"))

    def test_row_extra_key_rejected(self):
        self._reject(lambda r: r[2].__setitem__("extra", 1))

    def test_row_wrong_t_rejected(self):
        self._reject(lambda r: r[2].__setitem__("t", "nope"))

    def test_row_item_mismatch_rejected(self):
        self._reject(lambda r: r[2].__setitem__("item", 99))

    def test_row_pos_mismatch_rejected(self):
        self._reject(lambda r: r[2].__setitem__("pos", 5))

    def test_row_gold_mismatch_rejected(self):
        self._reject(lambda r: r[2].__setitem__("gold", 0))

    def test_row_am_wrong_type_rejected(self):
        self._reject(lambda r: r[2].__setitem__("am", "1"))

    def test_row_am_out_of_range_rejected(self):
        self._reject(lambda r: r[2].__setitem__("am", 4))

    def test_row_corr_wrong_type_rejected(self):
        self._reject(lambda r: r[2].__setitem__("corr", "1"))

    def test_row_corr_out_of_range_rejected(self):
        self._reject(lambda r: r[2].__setitem__("corr", 2))

    NUMERIC_FIELDS = ("nll", "glogit", "molo", "mgn", "amlogit", "logZ")

    def test_row_numeric_field_wrong_type_rejected(self):
        for field in self.NUMERIC_FIELDS:
            with self.subTest(field=field):
                self._reject(lambda r, field=field: r[2].__setitem__(
                    field, "0"))

    def test_row_tk_not_a_list_rejected(self):
        self._reject(lambda r: r[2].__setitem__("tk", 5))

    def test_row_tk_wrong_length_rejected(self):
        self._reject(lambda r: r[2].__setitem__(
            "tk", r[2]["tk"][:-1]))

    def test_row_tk_pair_wrong_shape_rejected(self):
        self._reject(lambda r: r[2]["tk"].__setitem__(0, [0, 0.1, 9]))

    def test_row_tk_pair_id_wrong_type_rejected(self):
        self._reject(lambda r: r[2]["tk"].__setitem__(0, ["0", 0.1]))

    def test_row_tk_pair_id_out_of_range_rejected(self):
        self._reject(lambda r: r[2]["tk"].__setitem__(0, [4, 0.1]))

    def test_row_tk_pair_val_wrong_type_rejected(self):
        self._reject(lambda r: r[2]["tk"].__setitem__(0, [0, "0.1"]))

    def test_row_tk_duplicate_ids_rejected(self):
        self._reject(lambda r: r[2]["tk"].__setitem__(0, list(r[2]["tk"][1])))


class CrossFieldInvariantTests(_GoldenFixture):
    """Invariants the engine's per-record emitter, `ablate_logit_record`
    (and the row-writer `ablate_logit_line` it calls), guarantees for
    every row it emits.

    - `nll >= 0`: `nll` is `-target_lp` (`ablate_logit_record`'s own
      `gnll=-target_lp`), and `target_lp` is `delta - logse` where
      `delta = lo[target] - r.max <= 0` (target's logit minus the row
      max) and `logse = log(sum_i exp(lo[i]-max)) >= log(1) = 0` (the
      max's own term contributes exp(0)=1 to that sum) -- the row-level
      helpers this emitter builds on (`logprob_row_checked`/
      `logprob_from_row_checked`). So `target_lp <= 0` always, hence
      `nll >= 0` always.
    - `corr == (am == gold)`: the emitter passes an `argmax==gold`
      comparison directly as the `corr` argument to `ablate_logit_line`,
      and `am` is that same argmax -- `corr` is never anything but that
      comparison's result.
    - `amlogit >= glogit`: `amlogit` is the row's own maximum logit and
      `glogit` is one particular entry of that same row, so it can
      never exceed the row's own maximum.

    Top-k ordering is deliberately NOT enforced: `tk` is unsorted on the
    wire by design (`logit_topk_select`, documented as "deliberately not
    a lowest-token-id tie rule").
    """

    def test_nll_negative_rejected(self):
        self._reject(lambda r: r[2].__setitem__("nll", -0.1))

    def test_nll_zero_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            records = self.golden_records()
            records[2]["nll"] = 0.0
            manifest, evidence, config = _write(
                root, self.MANIFEST, _serialize(records), self.CONFIG)
            ABLATE.validate_ablate_evidence(manifest, evidence, config)

    def test_corr_true_when_am_not_gold_rejected(self):
        # am=1 == gold=1 in the golden row, so corr must be 1; forcing 0
        # while leaving am/gold untouched breaks the agreement.
        self._reject(lambda r: r[2].__setitem__("corr", 0))

    def test_corr_false_when_am_equals_gold_rejected(self):
        # am=0 != gold=2 in row2 (index 3), so corr must be 0; forcing 1
        # breaks the agreement the other way.
        self._reject(lambda r: r[3].__setitem__("corr", 1))

    def test_amlogit_below_glogit_rejected(self):
        self._reject(lambda r: r[2].__setitem__("amlogit", 0.5))

    def test_amlogit_equal_to_glogit_accepted(self):
        # row1 already has amlogit == glogit == 1.0 (am == gold there);
        # confirm the boundary itself -- not just values strictly above
        # it -- is accepted.
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            records = self.golden_records()
            self.assertEqual(records[2]["amlogit"], records[2]["glogit"])
            manifest, evidence, config = _write(
                root, self.MANIFEST, _serialize(records), self.CONFIG)
            ABLATE.validate_ablate_evidence(manifest, evidence, config)


class TerminalRecordTests(_GoldenFixture):
    def test_missing_terminal_record_rejected(self):
        self._reject(lambda r: r.__delitem__(slice(6, None)))

    def test_terminal_not_a_dict_rejected(self):
        self._reject(lambda r: r.__setitem__(6, 5))

    def test_terminal_missing_key_rejected(self):
        self._reject(lambda r: r[6].pop("completed_items"))

    def test_terminal_extra_key_rejected(self):
        self._reject(lambda r: r[6].__setitem__("extra", 1))

    def test_terminal_completed_items_out_of_range_rejected(self):
        self._reject(lambda r: r[6].__setitem__("completed_items", 0))

    def test_terminal_completed_targets_out_of_range_rejected(self):
        self._reject(lambda r: r[6].__setitem__("completed_targets", 0))

    def test_terminal_wrong_t_rejected(self):
        self._reject(lambda r: r[6].__setitem__("t", "nope"))

    def test_terminal_manifest_sha256_mismatch_rejected(self):
        self._reject(lambda r: r[6].__setitem__("manifest_sha256", "1" * 64))

    def test_terminal_completed_items_mismatch_rejected(self):
        self._reject(lambda r: r[6].__setitem__("completed_items", 1))

    def test_terminal_completed_targets_mismatch_rejected(self):
        self._reject(lambda r: r[6].__setitem__("completed_targets", 1))

    def test_trailing_record_after_terminal_rejected(self):
        self._reject(lambda r: r.append(dict(r[6])))


class TruncationReplayAndMismatchBiteTests(_GoldenFixture):
    """Bite-style table close to the source's own producer-invariant
    checks, rebuilt on this module's literal golden fixture instead of
    an engine-produced one."""

    def test_named_mutations_all_refuse(self):
        cases = (
            ("missing_done", lambda r: r.__delitem__(6)),
            ("truncated_last_row", lambda r: r.__setitem__(
                5, {"t": "lg", "item": 2})),
            ("replayed_item_header", lambda r: r.insert(2, dict(r[1]))),
            ("duplicate_done", lambda r: r.append(dict(r[6]))),
            ("missing_target_row", lambda r: r.__delitem__(3)),
            ("wrong_gold_downstream", lambda r: r[3].__setitem__(
                "gold", 0)),
            ("header_digest_forged", lambda r: r[0].__setitem__(
                "manifest_sha256", "2" * 64)),
        )
        for name, mutate in cases:
            with self.subTest(name=name):
                self._reject(mutate)


class ManifestProofFramingTests(unittest.TestCase):
    """`_manifest_proof`'s non-canonical-text framing checks."""

    ARGS = (4, 4, 1, 5)  # vocab, n_layers, first_dense, n_experts

    def _reject(self, raw):
        with self.assertRaises(ABLATE.AblateEvidenceError):
            ABLATE._manifest_proof(raw, *self.ARGS)

    def test_empty_manifest_rejected(self):
        self._reject(b"")

    def _accept(self, raw):
        return ABLATE._manifest_proof(raw, *self.ARGS)

    # The engine accepts a manifest saved with CRLF endings and one whose last
    # line has no terminator, and digests the canonical form of either. A
    # checker that refused them would reject files the producer really ran.
    def test_manifest_missing_trailing_newline_accepted(self):
        self._accept(b"1 3 1 0 0 0 1 2")

    def test_manifest_with_crlf_endings_accepted(self):
        self._accept(b"1 3 1 0 0 0 1 2\r\n")

    def test_all_three_framings_bind_the_same_digest(self):
        canonical = self._accept(b"1 3 1 0 0 0 1 2\n")["sha256"]
        self.assertEqual(self._accept(b"1 3 1 0 0 0 1 2\r\n")["sha256"], canonical)
        self.assertEqual(self._accept(b"1 3 1 0 0 0 1 2")["sha256"], canonical)

    def test_carriage_return_inside_a_record_rejected(self):
        self._reject(b"1 3 1 0\r 0 0 1 2\n")

    def test_empty_line_inside_a_manifest_rejected(self):
        self._reject(b"1 3 1 0 0 0 1 2\n\n2 2 1 0 0 0 1\n")

    def test_manifest_with_nul_byte_rejected(self):
        self._reject(b"1 3 1 0 0 0 1 2\n\0")

    def test_manifest_non_ascii_line_rejected(self):
        self._reject("1 3 1 0 0 0 1 é\n".encode("utf-8"))

    def test_manifest_control_byte_breaks_grammar_not_framing(self):
        # A vertical tab embedded mid-line is not a canonical digit/space
        # byte; it must be caught by the integer-grammar check, not
        # silently absorbed as a line boundary bytes.splitlines() would
        # not treat it as one either way (see the dedicated probe below).
        self._reject(b"1 3\x0b1 0 0 0 1 2\n")

    def test_duplicate_item_id_across_lines_rejected(self):
        self._reject(b"1 2 1 0 0 0 1\n1 2 1 0 0 0 1\n")

    def test_records_are_split_on_newline_only(self):
        # The module now splits the canonical form on b"\n" rather than
        # calling splitlines(), so no other byte can become a record
        # boundary. bytes.splitlines() would additionally break on \r,
        # which the canonical form no longer contains but which a future
        # edit could reintroduce; splitting explicitly removes the
        # question. These bytes must therefore stay inside one record and
        # be caught by the integer-grammar check.
        for value in (0x0B, 0x0C, 0x1C, 0x1D, 0x1E):
            with self.subTest(byte=hex(value)):
                raw = b"1 3 1 0 0 0 1" + bytes([value]) + b"2\n"
                self._reject(raw)


class CanonicalManifestDigestTests(unittest.TestCase):
    """The canonical rule, pinned against the engine by a literal digest.

    `c/tests/test_ablate_mode.c` asserts the same 64 characters for the same
    manifest content. Two implementations that each only agreed with
    themselves would both pass their own suites while disagreeing in the
    field; a literal known answer on both sides is what rules that out.
    """

    RECORD = b"0 3 2 0 0 1 2 3\n"
    KNOWN = "c63a48c375b14ca60f26c7e3c5dd36b5929ffaf669a45511c93deee6e8bbd5ed"

    def test_known_answer_matches_the_engine(self):
        digest = hashlib.sha256(
            ABLATE.DOMAIN + engine_evidence.canonical_manifest_bytes(self.RECORD)
        ).hexdigest()
        self.assertEqual(digest, self.KNOWN)

    def test_every_accepted_framing_reaches_the_known_answer(self):
        for raw in (self.RECORD, b"0 3 2 0 0 1 2 3\r\n", b"0 3 2 0 0 1 2 3"):
            with self.subTest(raw=raw):
                digest = hashlib.sha256(
                    ABLATE.DOMAIN
                    + engine_evidence.canonical_manifest_bytes(raw)).hexdigest()
                self.assertEqual(digest, self.KNOWN)

    def test_canonicalization_refuses_what_the_engine_refuses(self):
        for raw in (b"", b"\n", b"a\n\nb\n", b"a\rb\n", b"a\0b\n"):
            with self.subTest(raw=raw):
                with self.assertRaises(engine_evidence.ManifestFormError):
                    engine_evidence.canonical_manifest_bytes(raw)


class ManifestProofFieldBoundaryTests(unittest.TestCase):
    """Per-field/per-cell/per-token bounds `_manifest_proof` enforces.

    Table built from `test_manifest_fixed_width_and_topology_c_python_parity`'s
    Python-side expectations (each `expected` value here is the same
    literal that method asserted, not something this module computed):
    that method also cross-checked each case against a C test binary
    this module does not build, so it is not this module's oracle to
    carry (flagged separately, not absorbed here).
    """

    def test_boundary_table(self):
        i32 = ABLATE._INT32_MAX
        i64 = ABLATE._INT64_MAX
        sixteen = " ".join(f"{layer} 0 -1" for layer in range(1, 17))
        cases = (
            ("baseline_min", b"0 2 1 0 0 0 1\n", 4, 4, 1, 8, True),
            ("fewer_than_five_fields", b"1 2\n", 4, 4, 1, 8, False),
            ("item_max", f"{i64} 2 1 0 0 0 1\n".encode(),
             4, 4, 1, 8, True),
            ("item_max_plus_1", f"{i64 + 1} 2 1 0 0 0 1\n".encode(),
             4, 4, 1, 8, False),
            ("item_min_minus_1", b"-1 2 1 0 0 0 1\n", 4, 4, 1, 8, False),
            ("T_min", b"7 2 1 0 0 0 1\n", 4, 4, 1, 8, True),
            ("T_below_min", b"7 1 1 0 0 0\n", 4, 4, 1, 8, False),
            ("T_max_incomplete", f"7 {i32} 1 0 0\n".encode(),
             4, 4, 1, 8, False),
            ("T_max_plus_1", f"7 {i32 + 1} 1 0 0\n".encode(),
             4, 4, 1, 8, False),
            ("prompt_max_incomplete", f"7 {i32} {i32 - 1} 0 0\n".encode(),
             4, 4, 1, 8, False),
            ("prompt_max_plus_1", f"7 {i32} {i32 + 1} 0 0\n".encode(),
             4, 4, 1, 8, False),
            ("prompt_min_minus_1", b"7 2 0 0 0 0 1\n",
             4, 4, 1, 8, False),
            ("mode_max", b"7 2 1 3 1 1 2 3 0 1\n",
             4, 4, 1, 8, True),
            ("mode_max_plus_1", b"7 2 1 4 1 1 2 -1 0 1\n",
             4, 4, 1, 8, False),
            ("cells_max", f"7 2 1 1 16 {sixteen} 0 1\n".encode(),
             4, 17, 1, 8, True),
            ("cells_max_plus_1", b"7 2 1 1 17 0 1\n",
             4, 17, 1, 8, False),
            ("nonbaseline_zero", b"7 2 1 1 0 0 1\n",
             4, 4, 1, 8, False),
            ("dense_layer", b"7 2 1 1 1 0 2 -1 0 1\n",
             4, 4, 1, 8, False),
            ("layer_min", b"7 2 1 1 1 0 2 -1 0 1\n",
             4, 4, 0, 8, True),
            ("layer_upper", b"7 2 1 1 1 3 2 -1 0 1\n",
             4, 4, 1, 8, True),
            ("layer_engine_max", b"7 2 1 1 1 127 2 -1 0 1\n",
             4, 128, 0, 8, True),
            ("layer_engine_max_plus_1", b"7 2 1 1 1 128 2 -1 0 1\n",
             4, 128, 0, 8, False),
            ("source_upper", b"7 2 1 1 1 1 7 -1 0 1\n",
             4, 4, 1, 8, True),
            ("source_min", b"7 2 1 1 1 1 0 -1 0 1\n",
             4, 4, 1, 8, True),
            ("source_engine_max", b"7 2 1 1 1 1 4095 -1 0 1\n",
             4, 4, 1, 4096, True),
            ("source_engine_max_plus_1", b"7 2 1 1 1 1 4096 -1 0 1\n",
             4, 4, 1, 4096, False),
            ("target_upper", b"7 2 1 3 1 1 2 7 0 1\n",
             4, 4, 1, 8, True),
            ("target_self_swap", b"7 2 1 3 1 1 2 2 0 1\n",
             4, 4, 1, 8, False),
            ("target_signed_min",
             f"7 2 1 3 1 1 2 {ABLATE._INT32_MIN} 0 1\n".encode(),
             4, 4, 1, 8, False),
            ("target_engine_max", b"7 2 1 3 1 1 0 4095 0 1\n",
             4, 4, 1, 4096, True),
            ("target_engine_max_plus_1", b"7 2 1 3 1 1 0 4096 0 1\n",
             4, 4, 1, 4096, False),
            ("target_max_plus_1",
             f"7 2 1 3 1 1 2 {i32 + 1} 0 1\n".encode(),
             4, 4, 1, 8, False),
            ("duplicate_source", b"7 2 1 1 2 1 2 -1 1 2 -1 0 1\n",
             4, 4, 1, 8, False),
            ("token_min", b"7 2 1 0 0 0 0\n", 1, 4, 1, 8, True),
            ("token_upper", b"7 2 1 0 0 0 16777215\n",
             1 << 24, 4, 1, 8, True),
            ("token_max_plus_1", b"7 2 1 0 0 0 16777216\n",
             1 << 24, 4, 1, 8, False),
            ("vocab_max", b"7 2 1 0 0 0 1\n",
             1 << 24, 4, 1, 8, True),
            ("vocab_max_plus_1", b"7 2 1 0 0 0 1\n",
             (1 << 24) + 1, 4, 1, 8, False),
            ("leading_zero_rejected", b"07 2 1 0 0 0 1\n",
             4, 4, 1, 8, False),
            ("plus_sign_rejected", b"+7 2 1 0 0 0 1\n",
             4, 4, 1, 8, False),
            ("double_space_rejected", b"7  2 1 0 0 0 1\n",
             4, 4, 1, 8, False),
            ("trailing_space_rejected", b"7 2 1 0 0 0 1 \n",
             4, 4, 1, 8, False),
        )
        for (name, raw, vocab, layers, first_dense, experts,
             expected) in cases:
            with self.subTest(name=name):
                if expected:
                    ABLATE._manifest_proof(
                        raw, vocab, layers, first_dense, experts)
                else:
                    with self.assertRaises(ABLATE.AblateEvidenceError):
                        ABLATE._manifest_proof(
                            raw, vocab, layers, first_dense, experts)


class ConfigIdentityTests(unittest.TestCase):
    CONFIG = (b'{"vocab_size":4,"num_hidden_layers":4,'
              b'"first_k_dense_replace":1,"n_routed_experts":5}\n')

    def test_engine_text_size_rejects_non_int_length(self):
        with self.assertRaises(ABLATE.AblateEvidenceError):
            ABLATE._checked_engine_text_size(True, "config")
        with self.assertRaises(ABLATE.AblateEvidenceError):
            ABLATE._checked_engine_text_size(1.0, "config")

    def test_engine_byte_limit_is_inclusive_and_enforced_both_sides(self):
        engine_limit = 256 << 20
        self.assertEqual(ABLATE._ENGINE_TEXT_MAX_BYTES, engine_limit)
        self.assertEqual(
            ABLATE._checked_engine_text_size(engine_limit, "config"),
            engine_limit)
        with self.assertRaises(ABLATE.AblateEvidenceError):
            ABLATE._checked_engine_text_size(engine_limit + 1, "config")

        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            ablate_config = root / "ablate-config.json"
            ablate_config.write_bytes(self.CONFIG)
            with mock.patch.object(
                    ABLATE, "_ENGINE_TEXT_MAX_BYTES", len(self.CONFIG)):
                identity = ABLATE._config_identity(ablate_config)
                self.assertEqual(identity["vocab"], 4)
                self.assertEqual(
                    identity["config_sha256"],
                    hashlib.sha256(self.CONFIG).hexdigest())
                ablate_config.write_bytes(self.CONFIG + b" ")
                with self.assertRaisesRegex(
                        ABLATE.AblateEvidenceError, "256 MiB"):
                    ABLATE._config_identity(ablate_config)

    def _reject(self, raw):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "config.json"
            path.write_bytes(raw)
            with self.assertRaises(ABLATE.AblateEvidenceError):
                ABLATE._config_identity(path)

    def test_empty_config_rejected(self):
        self._reject(b"")

    def test_invalid_json_config_rejected(self):
        self._reject(b"{not json}\n")

    def test_non_object_root_rejected(self):
        self._reject(b"[1,2,3]\n")

    def test_vocab_size_missing_rejected(self):
        self._reject(b'{"num_hidden_layers":4,"first_k_dense_replace":1,'
                     b'"n_routed_experts":5}\n')

    def test_vocab_size_out_of_range_rejected(self):
        self._reject(b'{"vocab_size":0,"num_hidden_layers":4,'
                     b'"first_k_dense_replace":1,"n_routed_experts":5}\n')

    def test_num_hidden_layers_out_of_range_rejected(self):
        self._reject(b'{"vocab_size":4,"num_hidden_layers":0,'
                     b'"first_k_dense_replace":1,"n_routed_experts":5}\n')

    def test_n_routed_experts_out_of_range_rejected(self):
        self._reject(b'{"vocab_size":4,"num_hidden_layers":4,'
                     b'"first_k_dense_replace":1,"n_routed_experts":0}\n')

    def test_first_k_dense_replace_out_of_range_rejected(self):
        self._reject(b'{"vocab_size":4,"num_hidden_layers":4,'
                     b'"first_k_dense_replace":5,"n_routed_experts":5}\n')


class JsonHelperTests(unittest.TestCase):
    def test_duplicate_json_key_rejected(self):
        with self.assertRaises(ABLATE.AblateEvidenceError):
            ABLATE._json_record(b'{"a":1,"a":2}', "record 1")

    def test_nan_constant_rejected(self):
        with self.assertRaises(ABLATE.AblateEvidenceError):
            ABLATE._json_record(b'{"nll":NaN}', "record 1")

    def test_infinity_constant_rejected(self):
        with self.assertRaises(ABLATE.AblateEvidenceError):
            ABLATE._json_record(b'{"nll":Infinity}', "record 1")

    def test_negative_infinity_constant_rejected(self):
        with self.assertRaises(ABLATE.AblateEvidenceError):
            ABLATE._json_record(b'{"nll":-Infinity}', "record 1")

    def test_non_ascii_bytes_rejected(self):
        with self.assertRaises(ABLATE.AblateEvidenceError):
            ABLATE._json_record(b"\xff", "record 1")

    def test_malformed_json_rejected(self):
        with self.assertRaises(ABLATE.AblateEvidenceError):
            ABLATE._json_record(b"{not json}", "record 1")


class FixedWidthHelperTests(unittest.TestCase):
    def test_bounded_int_rejects_bool_disguised_as_int(self):
        with self.assertRaises(ABLATE.AblateEvidenceError):
            ABLATE._bounded_int(True, "x", 1, 10)

    def test_int64_max_is_the_literal_signed_64_bit_bound(self):
        # Pinned by literal, not derived, so a future refactor of the
        # module's own (1 << 63) - 1 expression cannot silently drift.
        self.assertEqual(ABLATE._INT64_MAX, 9223372036854775807)

    def test_fixed_width_helpers_and_derived_count_boundaries(self):
        for value in (ABLATE._INT64_MIN, ABLATE._INT64_MAX):
            self.assertEqual(ABLATE._manifest_i64(str(value), 1), value)
        for value in (ABLATE._INT64_MIN - 1, ABLATE._INT64_MAX + 1):
            with self.assertRaises(ABLATE.AblateEvidenceError):
                ABLATE._manifest_i64(str(value), 1)
        for value in (ABLATE._INT32_MIN, ABLATE._INT32_MAX):
            self.assertEqual(ABLATE._bounded_int(
                value, "int32", ABLATE._INT32_MIN, ABLATE._INT32_MAX), value)
        for value in (ABLATE._INT32_MIN - 1, ABLATE._INT32_MAX + 1):
            with self.assertRaises(ABLATE.AblateEvidenceError):
                ABLATE._bounded_int(
                    value, "int32", ABLATE._INT32_MIN, ABLATE._INT32_MAX)
        self.assertEqual(
            ABLATE._count_add(0, ABLATE._INT64_MAX, "count"),
            ABLATE._INT64_MAX)
        self.assertEqual(
            ABLATE._count_add(ABLATE._INT64_MAX, 0, "count"),
            ABLATE._INT64_MAX)
        with self.assertRaises(ABLATE.AblateEvidenceError):
            ABLATE._count_add(ABLATE._INT64_MAX, 1, "count")
        for label in ("expected_items", "expected_targets",
                      "completed_items", "completed_targets"):
            self.assertEqual(
                ABLATE._bounded_int(1, label, 1, ABLATE._INT64_MAX), 1)
            self.assertEqual(ABLATE._bounded_int(
                ABLATE._INT64_MAX, label, 1, ABLATE._INT64_MAX),
                ABLATE._INT64_MAX)
            for value in (0, ABLATE._INT64_MAX + 1):
                with self.assertRaises(ABLATE.AblateEvidenceError):
                    ABLATE._bounded_int(value, label, 1, ABLATE._INT64_MAX)

    def test_retained_long_max_plus_one_artifact_is_incomplete(self):
        manifest_raw = b"9223372036854775808 2 1 0 0 0 1\n"
        digest = hashlib.sha256(_DOMAIN + manifest_raw).hexdigest()
        self.assertEqual(
            digest,
            "988a1cf2ddc812f38138e51eecfebb2ba0c9980e31b4c7183396716f114d6538")
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            config_raw = (b'{"vocab_size":2,"num_hidden_layers":4,'
                          b'"first_k_dense_replace":1,"n_routed_experts":8}\n')
            records = (
                {"t": "hdr", "schema": "coli-ablate/2", "vocab": 2,
                 "topk": 2, "n_layers": 4, "first_dense": 1,
                 "n_experts": 8,
                 "config_sha256": hashlib.sha256(config_raw).hexdigest(),
                 "manifest_sha256": digest,
                 "expected_items": 1, "expected_targets": 1},
                {"t": "ah", "item": 9223372036854775808, "mode": 0,
                 "ncells": 0, "T": 2, "n_prompt": 1, "cells": []},
                {"t": "lg", "item": 9223372036854775808, "pos": 0,
                 "gold": 1, "nll": 0, "glogit": 0, "molo": 0,
                 "mgn": 0, "am": 1, "amlogit": 0, "logZ": 0,
                 "corr": 1, "tk": [[1, 0], [0, -1]]},
                {"t": "done", "manifest_sha256": digest,
                 "completed_items": 1, "completed_targets": 1},
            )
            manifest, evidence, config = _write(
                root, manifest_raw, _serialize(records), config_raw)
            with self.assertRaises(ABLATE.AblateEvidenceError):
                ABLATE.validate_ablate_evidence(manifest, evidence, config)
            cli = _run_cli(manifest, evidence, config)
            self.assertNotEqual(cli.returncode, 0)
            self.assertIn(b"[ablate-evidence] INCOMPLETE:", cli.stderr)


if __name__ == "__main__":
    unittest.main()
