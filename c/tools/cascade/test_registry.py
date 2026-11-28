"""Cheap regression gates for evidence exchange; no model or GPU dependency."""
import hashlib
import json
from pathlib import Path
import unittest

OUT = Path(__file__).resolve().parents[3] / "docs/experiments/cascade"


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.extended = json.loads((OUT.parent / "dory-native/TRANSFER_REGISTRY.json").read_text())
        self.registry = self.extended["cascade"]
        self.results = json.loads((OUT / "sample-screen.json").read_text())

    def test_unique_versioned_complete_records(self):
        r = self.registry
        identities = [(c["id"], c["version"]) for c in r["cues"]]
        self.assertEqual(len(identities), len(set(identities)))
        for c in r["cues"]:
            self.assertTrue(set(r["required_cue_fields"]) <= c.keys())
            self.assertTrue(set(r["required_cascade_fields"]) <= c["cascade"].keys())

    def test_original_preserved(self):
        original = {k: v for k, v in self.extended.items() if k != "cascade"}
        canonical = json.dumps(original, sort_keys=True, separators=(",", ":")).encode()
        self.assertEqual(hashlib.sha256(canonical).hexdigest(), self.registry["source_canonical_sha256"])
        for c in self.registry["cues"]:
            self.assertEqual(c["provenance"]["registry_sha256"], self.registry["source_snapshot_sha256"])
            self.assertIn(c["original_mechanism_id"], [m["id"] for m in original["mechanisms"]])

    def test_no_false_model_promotion(self):
        self.assertFalse(self.results["promoted"])
        self.assertEqual(self.registry["coverage"]["full_model_runs"], 0)
        self.assertIsNone(self.registry["coverage"]["overall_model_gain"])
        self.assertEqual(len(self.results["cases"]), 18)
        for case in self.results["cases"]:
            self.assertTrue(case["exact_tokens_and_rng"])
            self.assertEqual(len(case["runs"]), 15)
            identities = [(x["tokens"], x["next_rng"]) for x in case["runs"]]
            self.assertTrue(all(i == identities[0] for i in identities))
        self.assertEqual(len(self.results["oracle"]), 3)
        for oracle in self.results["oracle"]:
            self.assertIn("1612 of 1612", oracle["output"])
            self.assertIn("PASSED", oracle["output"])

    def test_graph_acyclic(self):
        graph = {}
        for a, b in self.registry["dependency_edges"]:
            graph.setdefault(a, []).append(b)
        def visit(node, stack):
            self.assertNotIn(node, stack)
            for child in graph.get(node, []):
                visit(child, stack | {node})
        for node in graph:
            visit(node, set())

    def test_negative_feedback_retained(self):
        c = next(c for c in self.results["cases"] if
                 c["vocab"] == 248320 and c["shape"] == "peaked" and c["temperature"] == .7 and c["top_p"] == .9)
        self.assertLess(c["median_pair_rate_change_pct"], 0)
        self.assertTrue(any(x["kind"] == "negative_transfer" for x in self.registry["feedback"]))

    def test_edge_identity_and_results(self):
        edges = json.loads((OUT / "sample-edges.json").read_text())
        self.assertEqual(edges["generated_header_sha256"], self.results["sha256"]["generated_header"])
        self.assertTrue(edges["passed"])
        self.assertEqual([r["selector"] for r in edges["runs"]], [None, "1", "other"])
        for run in edges["runs"]:
            self.assertIn("609 checks, 0 failures", run["output"])

    def test_contrasts_recompute_from_raw_pairs(self):
        import statistics
        from summarize import paired
        contrasts = json.loads((OUT / "contrasts.json").read_text())
        self.assertTrue(contrasts["same_build_primary"])
        self.assertEqual(len(contrasts["cases"]), len(self.results["cases"]))
        for raw, derived in zip(self.results["cases"], contrasts["cases"]):
            self.assertEqual(derived["same_build_pairs_pct"], paired(raw, "off"))
            self.assertEqual(derived["same_build_median_pct"], statistics.median(paired(raw, "off")))


if __name__ == "__main__":
    unittest.main()
