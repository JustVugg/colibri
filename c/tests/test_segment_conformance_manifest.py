import json
import re
import unittest
from pathlib import Path

from family_registry import FAMILIES


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "tests" / "segment_conformance_manifest.json"
FIXTURE_SOURCE = ROOT / "tests" / "segment_conformance_fixtures.c"


class SegmentConformanceManifestTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        cls.entries = cls.manifest["families"]

    def test_is_all_registered_families_gate(self):
        self.assertEqual(self.manifest["version"], 1)
        self.assertEqual(
            self.manifest["release_policy"], "all_registered_families")
        manifest_ids = [entry["family_id"] for entry in self.entries]
        registry_ids = [family.id for family in FAMILIES]
        self.assertEqual(manifest_ids, registry_ids)
        self.assertEqual(len(manifest_ids), len(set(manifest_ids)))

    def test_every_entry_is_an_honest_contract_fixture(self):
        allowed_oracles = {"generated_tiny", "real_checkpoint"}
        for entry in self.entries:
            with self.subTest(family=entry["family_id"]):
                self.assertRegex(entry["family_id"], r"^[a-z0-9_]+$")
                self.assertTrue(entry["state_schema"].startswith("fixture/"))
                self.assertTrue(entry["state_components"])
                self.assertEqual(
                    len(entry["state_components"]),
                    len(set(entry["state_components"])))
                self.assertIs(entry["real_adapter_required"], True)
                self.assertIn(entry["oracle"]["kind"], allowed_oracles)
                self.assertTrue((ROOT / entry["oracle"]["path"]).is_file())

    def test_c_fixture_matrix_matches_manifest(self):
        source = FIXTURE_SOURCE.read_text(encoding="utf-8")
        schemas = set(re.findall(r'"(fixture/[a-z0-9_-]+-v[0-9]+)"', source))
        self.assertEqual(
            schemas, {entry["state_schema"] for entry in self.entries})
        adapters = set(re.findall(r"FIXTURE_ADAPTER\(([a-z0-9_]+)\)", source))
        adapters.discard("name")
        self.assertEqual(adapters, {entry["family_id"] for entry in self.entries})
        oracle_paths = set(re.findall(r'"(tools/make_[a-z0-9_-]+\.py)"', source))
        self.assertEqual(
            oracle_paths, {entry["oracle"]["path"] for entry in self.entries})


if __name__ == "__main__":
    unittest.main()
