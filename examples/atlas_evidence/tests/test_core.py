"""Checks independent mutations against exact evidence invariants."""
import copy
import json
import unittest

from examples.atlas_evidence.core import (assess, build_capture, digest, snapshot,
    source_watch_manifest, strict_json, validate_capture)


def fixture(text="A café policy: review lasts 30 days.", quote="review lasts 30 days."):
    source = snapshot("policy", text, path="sources/policy.txt")
    start = text.index(quote)
    claim = {"id": "claim-review", "text": "A candidate reading.", "source_id": "policy", "quote": quote, "start": start, "end": start + len(quote)}
    return build_capture([source], {"model": "fixture", "messages": []}, {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps({"claims": [claim]})}}]})


def change_claim(capture, **changes):
    changed = copy.deepcopy(capture)
    parsed = json.loads(changed["response"]["choices"][0]["message"]["content"])
    parsed["claims"][0].update(changes)
    changed["response"]["choices"][0]["message"]["content"] = json.dumps(parsed)
    return reseal(changed)


def reseal(capture):
    capture["sha256"] = digest({key: value for key, value in capture.items() if key != "sha256"})
    return capture


class CoreTests(unittest.TestCase):
    def test_unicode_codepoints_and_semantic_status(self):
        capture = fixture()
        report = assess(capture)
        self.assertTrue(report["passed"])
        self.assertEqual(report["semantic_status"], "unreviewed")
        self.assertEqual(report["claims"][0]["status"], "span_verified")
        self.assertFalse(assess(change_claim(capture, start=16))["passed"])

    def test_false_semantic_statement_can_pass_literal_check(self):
        capture = change_claim(fixture(), text="This policy is legally binding worldwide.")
        report = assess(capture)
        self.assertTrue(report["passed"])
        self.assertEqual(report["semantic_status"], "unreviewed")

    def test_fabrication_empty_boolean_and_bounds_rejected(self):
        for changes in ({"quote": "review lasts 90 days."}, {"quote": ""}, {"start": True}, {"end": 99999}, {"start": -1}, {"source_id": "missing"}):
            with self.subTest(changes=changes):
                self.assertFalse(assess(change_claim(fixture(), **changes))["passed"])

    def test_source_drift_and_removal_rejected(self):
        capture = fixture()
        changed = [snapshot("policy", capture["sources"][0]["text"].replace("30", "60"), path="sources/policy.txt")]
        self.assertFalse(assess(capture, changed)["passed"])
        self.assertFalse(assess(capture, [snapshot("different", "Another source.")])["passed"])
        self.assertFalse(assess(capture, [])["passed"])

    def test_policy_changes_check_without_changing_capture(self):
        capture = fixture()
        before = copy.deepcopy(capture)
        self.assertTrue(assess(capture)["passed"])
        self.assertFalse(assess(capture, min_quote_chars=100)["passed"])
        self.assertEqual(capture, before)
        with self.assertRaises(ValueError):
            assess(capture, min_quote_chars=True)

    def test_source_and_capture_tampering(self):
        capture = fixture()
        capture["sources"][0]["text"] += " Changed"
        with self.assertRaises(ValueError):
            validate_capture(capture)
        capture = fixture()
        capture["model_revision"] = "modified"
        with self.assertRaises(ValueError):
            validate_capture(capture)

    def test_strict_json(self):
        for raw in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}', '{"a":1e999}', '{"a":1,}', '"\\ud800"'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                strict_json(raw)

    def test_response_failures_are_recorded(self):
        for response in ({}, {"choices": []}, {"choices": [{"finish_reason": "length", "message": {"content": "{}"}}]}, {"choices": [{"finish_reason": "stop", "message": {"content": "{}", "tool_calls": [{"id": "x"}]}}]}, {"choices": [{"finish_reason": "stop", "message": {"content": '{"claims":[]}'}}]}, {"choices": [{"finish_reason": "stop", "message": {"content": '{"claims":[],"claims":[]}'}}]}):
            capture = fixture()
            capture["response"] = response
            report = assess(reseal(capture))
            self.assertFalse(report["passed"])
            self.assertTrue(report["errors"])

    def test_duplicate_claims_and_identifier_collision(self):
        capture = fixture()
        parsed = json.loads(capture["response"]["choices"][0]["message"]["content"])
        parsed["claims"].append(copy.deepcopy(parsed["claims"][0]))
        capture["response"]["choices"][0]["message"]["content"] = json.dumps(parsed)
        self.assertFalse(assess(reseal(capture))["passed"])
        self.assertFalse(assess(change_claim(fixture(), id="policy"))["passed"])

    def test_bounded_sources_and_unsafe_paths(self):
        with self.assertRaises(ValueError):
            snapshot("policy", "é" * 200000)
        for path in ("../secret.txt", "/etc/passwd", ".git/config", "x/.GIT/config", "c:\\secret", " a.txt", "a//b"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                snapshot("policy", "text", path=path)

    def test_export_metadata_respects_utf16_bounds(self):
        with self.assertRaises(ValueError):
            snapshot("policy", "text", title="🐦" * 600)
        self.assertEqual(snapshot("policy", "text", title="🐦" * 512)["title"], "🐦" * 512)
        self.assertFalse(assess(change_claim(fixture(), text="🐦" * 5000))["passed"])

    def test_source_urls_reject_invalid_hosts_and_ports(self):
        for url in ("https://example.com:invalid/path", "https://example.com:99999/path", "https://999.999.999.999/x", "https://bad_host.example/x", "https://user:secret@example.com/x"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                snapshot("policy", "text", url=url)
        for url in ("https://example.com:8443/path", "http://[::1]:8000/source", "https://例え.jp/path"):
            with self.subTest(url=url):
                self.assertEqual(snapshot("policy", "text", url=url)["url"], url)

    def test_export_has_context_only_and_no_review(self):
        capture = fixture()
        manifest = source_watch_manifest(capture, assess(capture))
        claim = manifest["claims"][0]
        self.assertEqual(claim["evidence"][0]["relation"], "context")
        self.assertNotIn("review", claim)

    def test_export_rejects_forged_report(self):
        capture = fixture()
        original = assess(capture)
        for edit in (lambda report: report.update(min_quote_chars=100), lambda report: report["current_source_hashes"].update(policy="0" * 64), lambda report: report["claims"][0]["candidate"].update(text="Forged text"), lambda report: report.update(passed=False)):
            report = copy.deepcopy(original)
            edit(report)
            with self.assertRaises(ValueError):
                source_watch_manifest(capture, report)


if __name__ == "__main__":
    unittest.main()
