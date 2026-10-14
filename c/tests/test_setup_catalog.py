"""The setup's model catalog and its recommendation, on mocked machines.

The catalog names registry families; the registry owns engines, build targets
and accelerator support. The first class holds the two together, so a family
renamed in the registry cannot leave the setup offering an engine that does not
exist. The second runs the recommendation on machines described by numbers.
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import setup_catalog  # noqa: E402
from family_registry import family_by_id  # noqa: E402

GB = 10**9


def recommended(rows):
    picks = [r["entry"].id for r in rows if r["recommended"]]
    return picks[0] if picks else None


class CatalogAgreesWithTheRegistry(unittest.TestCase):
    def test_every_entry_names_a_registry_family(self):
        for entry in setup_catalog.CATALOG:
            with self.subTest(entry=entry.id):
                family = family_by_id(entry.family)
                self.assertTrue(family.build_target)
                if entry.modality != "text":
                    self.assertEqual(family.modality, entry.modality)

    def test_ids_are_unique_and_numbers_sane(self):
        ids = [entry.id for entry in setup_catalog.CATALOG]
        self.assertEqual(len(ids), len(set(ids)))
        for entry in setup_catalog.CATALOG:
            with self.subTest(entry=entry.id):
                self.assertGreater(entry.disk_gb, 0)
                self.assertLessEqual(entry.ram_min_gb, entry.ram_good_gb)
                self.assertLess(entry.dense_gb, entry.ram_min_gb)
                self.assertIn(entry.size_class, ("small", "large"))
                self.assertRegex(entry.repo, r"^[\w.-]+/[\w.-]+$")
                for step in entry.post_install:
                    self.assertTrue((Path(__file__).resolve().parent.parent / "tools"
                                     / f"{step[0]}.py").is_file(), step)

    def test_lookup(self):
        self.assertEqual(setup_catalog.by_id("qwen36-35b").family, "qwen36")
        with self.assertRaises(KeyError):
            setup_catalog.by_id("no-such-model")


class Recommendation(unittest.TestCase):
    def test_laptop_with_27_gb_gets_the_model_that_runs_from_ram(self):
        rows = setup_catalog.recommend(27 * GB, 800 * GB)
        self.assertEqual(recommended(rows), "qwen36-35b")
        status = {r["entry"].id: r["status"] for r in rows}
        self.assertEqual(status["mimo-v2.6-flash"], "needs-ram")
        self.assertEqual(status["glm-5.2"], "good")
        # fitting rows first, the recommendation at the top
        self.assertTrue(rows[0]["recommended"])
        fits = [r["fits"] for r in rows]
        self.assertEqual(fits, sorted(fits, reverse=True))

    def test_small_ram_takes_the_smaller_model(self):
        rows = setup_catalog.recommend(9 * GB, 200 * GB)
        self.assertEqual(recommended(rows), "qwen3-coder-30b")

    def test_tight_ram_prefers_the_most_capable_small_model_that_runs(self):
        rows = setup_catalog.recommend(16 * GB, 200 * GB)
        self.assertEqual(recommended(rows), "qwen36-35b")
        self.assertEqual({r["entry"].id: r["status"] for r in rows}["qwen36-35b"], "tight")

    def test_nothing_fits(self):
        rows = setup_catalog.recommend(4 * GB, 1000 * GB)
        self.assertIsNone(recommended(rows))
        self.assertFalse(any(r["fits"] for r in rows))

    def test_disk_decides_too(self):
        # 22 GB free: room for the 19.4 GB coder (plus 2% and 2 GB of slack),
        # not for the 23.1 GB Qwen3.6, whatever the RAM.
        rows = setup_catalog.recommend(64 * GB, 22 * GB)
        status = {r["entry"].id: r["status"] for r in rows}
        self.assertEqual(status["qwen36-35b"], "needs-disk")
        self.assertEqual(status["glm-5.2"], "needs-disk")
        self.assertEqual(recommended(rows), "qwen3-coder-30b")

    def test_a_partial_download_counts_toward_the_disk(self):
        rows = setup_catalog.recommend(64 * GB, 10 * GB, downloaded={"qwen36-35b": 20 * GB})
        self.assertEqual({r["entry"].id: r["status"] for r in rows}["qwen36-35b"], "good")

    def test_large_models_are_the_default_only_when_no_small_one_fits(self):
        with mock.patch.object(setup_catalog, "CATALOG",
                               tuple(e for e in setup_catalog.CATALOG if e.size_class == "large")):
            rows = setup_catalog.recommend(27 * GB, 800 * GB)
        self.assertEqual(recommended(rows), "deepseek-v4-flash-reap")   # smallest download

    def test_the_image_model_is_offered_never_the_default(self):
        with mock.patch.object(setup_catalog, "CATALOG",
                               tuple(e for e in setup_catalog.CATALOG if e.modality == "image")):
            rows = setup_catalog.recommend(64 * GB, 800 * GB)
        self.assertTrue(rows[0]["fits"])
        self.assertIsNone(recommended(rows))

    def test_platform_limits(self):
        rows = setup_catalog.recommend(64 * GB, 800 * GB, os_name="darwin", machine="x86_64")
        self.assertEqual({r["entry"].id: r["status"] for r in rows}["deepseek-v4-flash"], "unsupported")
        rows = setup_catalog.recommend(64 * GB, 800 * GB, os_name="win32", machine="amd64")
        self.assertEqual({r["entry"].id: r["status"] for r in rows}["deepseek-v4-flash"], "good")

    def test_reasons_explain_ram_as_dense_plus_cache(self):
        rows = setup_catalog.recommend(8 * GB, 800 * GB)
        reason = {r["entry"].id: r["reason"] for r in rows}["mimo-v2.6-flash"]
        self.assertIn("dense part", reason)
        self.assertIn("expert cache", reason)

    def test_as_dict_is_json(self):
        rows = setup_catalog.recommend(27 * GB, 800 * GB)
        data = [setup_catalog.as_dict(r) for r in rows]
        json.dumps(data)
        self.assertEqual(sum(1 for d in data if d["recommended"]), 1)
        self.assertEqual(data[0]["runs"], "from RAM")


class ExtraCatalog(unittest.TestCase):
    def test_entries_from_the_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "extra.json")
            with open(path, "w") as handle:
                json.dump([{"id": "mirror-qwen", "family": "qwen36", "name": "Mirror",
                            "repo": "me/mirror", "disk_gb": 1, "ram_min_gb": 1, "ram_good_gb": 2,
                            "dense_gb": 0.5, "rank": 50, "size_class": "small",
                            "summary": "test", "exclude": ["*.md"], "unknown_key": 1}], handle)
            with mock.patch.dict(os.environ, {"COLI_SETUP_CATALOG": path}):
                entry = setup_catalog.by_id("mirror-qwen")
                rows = setup_catalog.recommend(27 * GB, 800 * GB)
        self.assertEqual(entry.exclude, ("*.md",))
        self.assertEqual(recommended(rows), "mirror-qwen")


class Versions(unittest.TestCase):
    def test_version_order(self):
        v = setup_catalog.version_tuple
        self.assertLess(v("v1.12.1"), v("1.12.2"))
        self.assertLess(v("1.12.9"), v("v1.13.0"))
        self.assertEqual(v("1.12"), (1, 12, 0))


if __name__ == "__main__":
    unittest.main()
