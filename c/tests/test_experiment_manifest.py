import copy
import unittest

from experiment_manifest import validate


def run(config, speeds):
    return {
        "config": config,
        "samples": {"tok_s": speeds},
        "median_tok_s": sorted(speeds)[1],
        "quality": {"method": "token-exact oracle", "passed": True},
        "evidence": {"uri": "https://example.invalid/raw.log",
                     "sha256": "ab" * 32},
    }


def manifest():
    return {
        "version": 1,
        "hypothesis": "two loader lanes improve cold decode",
        "commit": "12" * 20,
        "model": "GLM-5.2 int4",
        "command": "NGEN=16 PROF=1 ./coli run ...",
        "prompt_hash": "sha256:example",
        "hardware": {"cpu": "example", "ram": "128 GB",
                     "storage": "NVMe ext4", "os": "Linux"},
        "warmup_runs": 1,
        "changed_variables": ["PIPE_WORKERS"],
        "baseline": run({"PIPE_WORKERS": "1", "DIRECT": "1"}, [1.0, 1.1, 1.2]),
        "trial": run({"PIPE_WORKERS": "2", "DIRECT": "1"}, [1.2, 1.3, 1.4]),
        "outcome": "improvement",
    }


class ExperimentManifestTest(unittest.TestCase):
    def test_accepts_reproducible_one_variable_record(self):
        result = validate(manifest())
        self.assertEqual(result["variable"], "PIPE_WORKERS")

    def test_rejects_hidden_second_variable(self):
        record = manifest()
        record["trial"]["config"]["DIRECT"] = "0"
        with self.assertRaisesRegex(ValueError, "config diff"):
            validate(record)

    def test_rejects_claimed_median_not_backed_by_samples(self):
        record = manifest()
        record["trial"]["median_tok_s"] = 9.9
        with self.assertRaisesRegex(ValueError, "sample median"):
            validate(record)

    def test_rejects_missing_quality_gate(self):
        record = manifest()
        record["trial"]["quality"]["passed"] = False
        with self.assertRaisesRegex(ValueError, "passed must be true"):
            validate(record)

    def test_rejects_unhashed_raw_evidence(self):
        record = copy.deepcopy(manifest())
        record["baseline"]["evidence"]["sha256"] = "unknown"
        with self.assertRaisesRegex(ValueError, "64 hex"):
            validate(record)


if __name__ == "__main__":
    unittest.main()
