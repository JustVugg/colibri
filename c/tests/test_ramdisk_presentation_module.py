"""Direct seams for the dependency-free headless presentation module."""

import contextlib
import io
import sys
import unittest
from pathlib import Path


C_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(C_DIR))

from ramdisk_support import presentation  # noqa: E402


def sample_plan(topology="interleaved"):
    mounts = (
        [{"node": None, "path": "/mnt/colibri-ram/shared"}]
        if topology == "interleaved"
        else [
            {"node": 0, "path": "/mnt/colibri-ram/node0"},
            {"node": 1, "path": "/mnt/colibri-ram/node1"},
        ]
    )
    staged = 4 << 30
    return {
        "topology": topology,
        "mode": "full",
        "model": {"path": "/models/example"},
        "hardware": {"online_nodes": [0, 1]},
        "placement": {
            "memory_nodes": [0, 1],
            "memory_node_list": "0-1",
            "cpu_list": "0-3",
        },
        "staging": {
            "selected_shards": ["model-00001.safetensors"],
            "direct_mapped_expert_count": 2,
            "staged_bytes": staged,
        },
        "mounts": mounts,
        "reserve": {
            "total_required_bytes": 6 << 30,
            "available_bytes": 8 << 30,
        },
        "warnings": [],
        "blockers": [],
    }


class PresentationModuleTest(unittest.TestCase):
    def test_shared_and_replica_summaries_preserve_operator_consequences(self):
        shared = presentation._placement_summary(sample_plan())
        replicated = presentation._placement_summary(sample_plan("per-node"))

        self.assertEqual(shared["copy_count"], 1)
        self.assertEqual(shared["engine_count"], 1)
        self.assertEqual(shared["endpoints"], "port 8000")
        self.assertEqual(replicated["copy_count"], 2)
        self.assertEqual(replicated["engine_count"], 2)
        self.assertEqual(replicated["endpoints"], "ports 8000, 8001")
        self.assertIn("replication, not model sharding", replicated["explanation"])

    def test_accelerator_review_is_a_pure_projection(self):
        plan = sample_plan()
        plan["managed_accelerator"] = {
            "mode": "cuda",
            "layout": "dense-attention-sharded",
            "devices": [
                {"index": 0, "name": "GPU 0", "numa_node": 0},
                {"index": 2, "name": "GPU 2", "numa_node": 1},
            ],
        }
        plan["accelerator_projection"] = {
            "dense_gpu_bytes": 2 << 30,
            "expert_headroom_bytes": 3 << 30,
            "vram_reserve_per_device_bytes": 1 << 30,
        }

        review = presentation._accelerator_review(plan)

        self.assertEqual(review["indices"], "0,2")
        self.assertEqual(review["layout"], "dense-attention-sharded")
        self.assertEqual(review["dense_gpu_gib"], 2.0)
        self.assertEqual(review["expert_headroom_gib"], 3.0)
        self.assertEqual(review["reserve_per_device_gib"], 1.0)

    def test_human_plan_uses_only_headless_projection(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            presentation._human_plan(sample_plan())

        rendered = output.getvalue()
        self.assertIn("RAM-disk plan: Single shared model", rendered)
        self.assertIn("endpoints after start: port 8000", rendered)
        self.assertIn("NUMA memory nodes: 0-1", rendered)

    def test_human_status_sanitizes_unknown_recovery_fields(self):
        secret = "secret-recovery-nonce"
        report = {
            "present": True,
            "state": "error",
            "mounts": [],
            "processes": [],
            "recovery": {
                "operation": "prepare",
                "state": "attention-required",
                "retained_mounts": ["/mnt/colibri-ram/shared"],
                "nonce": secret,
            },
        }
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            presentation._human_status(report)

        rendered = output.getvalue()
        self.assertIn("recovery: prepare / attention-required", rendered)
        self.assertIn("retained mount: /mnt/colibri-ram/shared", rendered)
        self.assertNotIn(secret, rendered)


if __name__ == "__main__":
    unittest.main()
