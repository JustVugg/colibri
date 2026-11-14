"""Integration tests using the actual optional Reweave engine and SQLite store."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from examples.atlas_evidence.core import build_capture, snapshot
from examples.atlas_evidence.replay import make_stages, run_review


class MissingDependencyTests(unittest.TestCase):
    def test_missing_reweave_is_actionable_import_error(self):
        import builtins
        from examples.atlas_evidence.replay import _dependency
        original_import = builtins.__import__

        def without_reweave(name, *args, **kwargs):
            if name == "reweave" or name.startswith("reweave."):
                raise ModuleNotFoundError("No module named 'reweave'", name="reweave")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=without_reweave):
            with self.assertRaisesRegex(ImportError, "requirements.txt"):
                _dependency()


@unittest.skipUnless(importlib.util.find_spec("reweave"), "optional Atlas Reweave is not installed")
class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = Path(self.temp.name) / "review.sqlite3"
        self.text = "Invented policy. Reports are reviewed annually. End."
        self.quote = "Reports are reviewed annually."
        self.source = snapshot("policy", self.text, title="Invented policy",
                               path="sources/policy.txt")
        start = self.text.index(self.quote)
        self.claim = {"id": "review-cycle", "text": "The policy specifies annual review.",
                      "source_id": "policy", "quote": self.quote,
                      "start": start, "end": start + len(self.quote)}
        self.capture = self._capture(self.claim)

    def _capture(self, claim):
        request = {"model": "fixture", "messages": [{"role": "user", "content": "Extract candidate evidence."}],
                   "stream": False}
        response = {"model": "fixture", "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": json.dumps({"claims": [claim]})}}]}
        return build_capture([self.source], request, response, mode="fixture",
                             model_revision="fixture-v1", engine_revision="fixture-v1")

    def test_first_review_and_repeat_have_real_durable_reuse(self):
        with patch("socket.create_connection", side_effect=AssertionError("No network in replay")), \
             patch("examples.atlas_evidence.client.generate", side_effect=AssertionError("No inference in replay")):
            first = run_review(self.capture, self.store)
            repeated = run_review(self.capture, self.store)
        self.assertEqual(first["report_receipt"]["status"], "accepted")
        self.assertEqual(first["stats"]["producer_calls"], 3)
        self.assertEqual(repeated["stats"]["producer_calls"], 0)
        self.assertEqual(first["report_receipt"]["work_key"], repeated["report_receipt"]["work_key"])
        self.assertEqual(repeated["model_requests"], 0)
        self.assertIsNotNone(first["bundle"])
        self.assertEqual(len(first["bundle"]["works"]), 3)

    def test_changed_policy_rejects_saved_evidence_and_blocks_cached_report(self):
        first = run_review(self.capture, self.store)
        stricter = run_review(self.capture, self.store, min_quote_chars=len(self.quote) + 1)
        for field in ("capture_receipt", "evidence_receipt", "report_receipt"):
            self.assertEqual(first[field]["work_key"], stricter[field]["work_key"])
        self.assertNotEqual(first["evidence_receipt"]["judgement_key"],
                            stricter["evidence_receipt"]["judgement_key"])
        self.assertEqual(stricter["capture_receipt"]["status"], "accepted")
        self.assertEqual(stricter["evidence_receipt"]["status"], "rejected")
        self.assertEqual(stricter["report_receipt"]["status"], "blocked")
        self.assertNotIn("output", stricter["report_receipt"])
        self.assertIsNone(stricter["bundle"])
        self.assertEqual(stricter["stats"]["producer_calls"], 0)
        restored = run_review(self.capture, self.store)
        self.assertEqual(restored["report_receipt"]["status"], "accepted")
        self.assertEqual(restored["stats"]["producer_calls"], 0)

    def test_source_drift_changes_judgement_without_new_work(self):
        first = run_review(self.capture, self.store)
        changed = snapshot("policy", self.text.replace("annually", "quarterly"),
                           title="Invented policy", path="sources/policy.txt")
        drifted = run_review(self.capture, self.store, current_sources=[changed])
        self.assertEqual(first["evidence_receipt"]["work_key"], drifted["evidence_receipt"]["work_key"])
        self.assertEqual(drifted["evidence_receipt"]["status"], "rejected")
        self.assertEqual(drifted["report_receipt"]["status"], "blocked")
        self.assertEqual(drifted["stats"]["producer_calls"], 0)
        self.assertFalse(drifted["current_review"]["passed"])

    def test_direct_revalidation_blocks_export_after_policy_change(self):
        from reweave import Engine, ReweaveError
        first = run_review(self.capture, self.store)
        with Engine(self.store, make_stages([self.source], len(self.quote) + 1),
                    scope="colibri-atlas-evidence-v1") as engine:
            receipt = engine.revalidate(first["report_receipt"]["work_key"])
            self.assertEqual(receipt.status, "blocked")
            with self.assertRaises(ReweaveError):
                engine.export([receipt.work_key])
            self.assertEqual(engine.stats()["producer_calls"], 0)

    def test_capture_larger_than_default_reweave_input_limit_is_supported(self):
        source = snapshot("policy", self.text + " fixture" * 9000,
                          title="Invented policy", path="sources/policy.txt")
        captured = build_capture([source], self.capture["request"], self.capture["response"],
                                 mode="fixture", model_revision="fixture-v1", engine_revision="fixture-v1")
        self.assertGreater(len(json.dumps(captured).encode()), 65536)
        result = run_review(captured, self.store)
        self.assertEqual(result["report_receipt"]["status"], "accepted")

    def test_fabricated_quote_cannot_export_evidence(self):
        claim = dict(self.claim, quote="Reports are never reviewed.")
        result = run_review(self._capture(claim), self.store)
        self.assertEqual(result["evidence_receipt"]["status"], "rejected")
        self.assertEqual(result["report_receipt"]["status"], "blocked")
        self.assertIsNone(result["bundle"])

    def test_export_detects_modified_evidence(self):
        from reweave.bundle import verify_bundle
        from reweave import StoreCorrupt
        result = run_review(self.capture, self.store)
        changed = deepcopy(result["bundle"])
        changed["works"][0]["output"] = {"substituted": True}
        with self.assertRaises(StoreCorrupt):
            verify_bundle(changed)

    def test_capture_tampering_fails_before_store_is_created(self):
        changed = deepcopy(self.capture)
        changed["sources"][0]["text"] += " Added text."
        with self.assertRaises(ValueError):
            run_review(changed, self.store)
        self.assertFalse(self.store.exists())

    def test_current_snapshot_does_not_borrow_mutable_caller_state(self):
        from reweave import Engine
        current = [deepcopy(self.source)]
        stages = make_stages(current)
        current[0]["text"] = "Externally mutated after registration"
        with Engine(self.store, stages) as engine:
            imported = engine.run("atlas.capture", self.capture)
            evidence = engine.run("atlas.evidence", {}, parents=[imported.work_key])
        self.assertEqual(evidence.status, "accepted")


if __name__ == "__main__":
    unittest.main()
