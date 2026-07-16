import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from resource_plan import (
    GB,
    analyze_model,
    build_plan,
    environment_for_plan,
    format_plan,
    memory_available,
)


def write_shard(path, tensors):
    offset = 0
    header = {}
    payload = b""
    for name, size in tensors:
        header[name] = {"dtype": "U8", "shape": [size], "data_offsets": [offset, offset + size]}
        payload += b"\0" * size
        offset += size
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)


class ResourcePlanTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.model = Path(self.tmp.name)
        (self.model / "config.json").write_text(json.dumps({
            "num_hidden_layers": 2,
            "n_routed_experts": 2,
            "kv_lora_rank": 4,
            "qk_rope_head_dim": 2,
            "qk_nope_head_dim": 3,
            "v_head_dim": 5,
            "num_attention_heads": 2,
        }))
        write_shard(self.model / "model.safetensors", [
            ("model.embed_tokens.weight", 100),
            ("model.layers.0.self_attn.q_a_proj.weight", 200),
            ("model.layers.1.mlp.experts.0.gate_proj.weight", 30),
            ("model.layers.1.mlp.experts.0.up_proj.weight", 30),
            ("model.layers.1.mlp.experts.1.gate_proj.weight", 30),
            ("model.layers.1.mlp.experts.1.up_proj.weight", 30),
        ])

    def tearDown(self):
        self.tmp.cleanup()

    def test_analyzes_dense_and_expert_storage(self):
        info = analyze_model(self.model)
        self.assertEqual(info["dense_bytes"], 300)
        self.assertEqual(info["expert_bytes"], 120)
        self.assertEqual(info["expert_count"], 2)
        self.assertEqual(info["per_cap_bytes"], 60)

    def test_memory_available_is_positive(self):
        # Regression: on native Windows CPython, /proc/meminfo does not exist,
        # so the Linux-only path returned 0 and the expert cache was sized to
        # 0 slots/layer. The value must be a sane positive number of bytes.
        self.assertGreater(memory_available(), 0)

    def test_builds_bounded_three_tier_plan(self):
        gpus = [{"index": 0, "name": "test-gpu", "total_bytes": 12 * GB,
                 "free_bytes": 10 * GB}]
        plan = build_plan(self.model, ram_gb=16, context=32, vram_gb=20,
                          available_memory=32 * GB, available_disk=100 * GB, gpus=gpus,
                          physical_cpus=24)
        self.assertEqual(plan["version"], 2)
        self.assertEqual(plan["policy"]["name"], "quality")
        self.assertEqual(plan["cpu"]["physical_cores"], 24)
        self.assertTrue(plan["policy"]["preserve_quantization"])
        self.assertFalse(plan["tiers"]["vram"]["requires_host_backing"])
        self.assertEqual(plan["tiers"]["ram"]["budget_bytes"], 16 * GB)
        self.assertLessEqual(plan["tiers"]["vram"]["budget_bytes"], 8 * GB)
        self.assertIn("clamped", plan["warnings"][0])
        self.assertIn("0:test-gpu", format_plan(plan))

    def test_filters_requested_devices(self):
        gpus = [{"index": 0, "name": "a", "total_bytes": 8 * GB, "free_bytes": 8 * GB}]
        plan = build_plan(self.model, available_memory=16 * GB, available_disk=1,
                          gpus=gpus, gpu_indices=[1])
        self.assertEqual(plan["tiers"]["vram"]["devices"], [])
        self.assertIn("not detected", plan["warnings"][0])

    def test_cli_emits_versioned_json(self):
        cli = Path(__file__).parents[1] / "coli"
        run = subprocess.run([
            sys.executable, str(cli), "plan", "--model", str(self.model),
            "--gpu", "none", "--json",
        ], text=True, capture_output=True, check=True)
        plan = json.loads(run.stdout)
        self.assertEqual(plan["version"], 2)
        self.assertEqual(plan["model"]["expert_count"], 2)

    def test_applies_plan_without_overriding_explicit_settings(self):
        gpus = [
            {"index": 0, "name": "a", "total_bytes": 12 * GB, "free_bytes": 10 * GB},
            {"index": 1, "name": "b", "total_bytes": 12 * GB, "free_bytes": 10 * GB},
        ]
        plan = build_plan(self.model, ram_gb=16, available_memory=32 * GB,
                          available_disk=1, gpus=gpus)
        env = environment_for_plan(plan, {"RAM_GB": "12", "PIN": "stats.txt",
                                               "COLI_GPUS": "1"})
        self.assertEqual(env["RAM_GB"], "12")
        self.assertEqual(env["COLI_CUDA"], "1")
        self.assertEqual(env["COLI_GPUS"], "1")
        self.assertEqual(env["OMP_NUM_THREADS"], str(plan["cpu"]["physical_cores"]))
        if sys.platform == "win32":
            # MinGW libgomp: niente affinity su Windows, le chiavi non vanno emesse
            self.assertNotIn("OMP_PROC_BIND", env)
            self.assertNotIn("OMP_PLACES", env)
        else:
            self.assertEqual(env["OMP_PROC_BIND"], "spread")
            self.assertEqual(env["OMP_PLACES"], "cores")
        self.assertEqual(env["PIN_GB"], env["CUDA_EXPERT_GB"])

        explicit_threads = environment_for_plan(plan, {"OMP_NUM_THREADS": "7",
                                                        "OMP_PROC_BIND": "close"})
        self.assertEqual(explicit_threads["OMP_NUM_THREADS"], "7")
        self.assertEqual(explicit_threads["OMP_PROC_BIND"], "close")
    def test_cpu_binary_does_not_apply_gpu_tier(self):
        plan = build_plan(self.model, available_memory=16 * GB, available_disk=1,
                          gpus=[{"index": 0, "name": "a", "total_bytes": 8 * GB,
                                 "free_bytes": 8 * GB}])
        env = environment_for_plan(plan, cuda_enabled=False)
        self.assertIn("RAM_GB", env)
        self.assertNotIn("COLI_CUDA", env)
        disabled = environment_for_plan(plan, {"COLI_CUDA": "0"}, cuda_enabled=True)
        self.assertNotIn("COLI_GPU", disabled)
        self.assertNotIn("CUDA_EXPERT_GB", disabled)

    def test_rejects_unknown_policy_and_marks_experimental_policy(self):
        with self.assertRaisesRegex(ValueError, "unknown policy"):
            build_plan(self.model, available_memory=16 * GB, available_disk=1,
                       gpus=[], policy="fast-ish")
        plan = build_plan(self.model, available_memory=16 * GB, available_disk=1,
                          gpus=[], policy="experimental-fast")
        self.assertFalse(plan["policy"]["quality_preserving"])
        self.assertFalse(plan["policy"]["preserve_router"])

    def test_balanced_policy_enables_lossless_live_repin(self):
        plan = build_plan(self.model, available_memory=16 * GB, available_disk=1,
                          gpus=[], policy="balanced")
        env = environment_for_plan(plan)
        self.assertEqual(env["COLI_POLICY"], "balanced")
        self.assertEqual(env["REPIN"], "64")
        explicit = environment_for_plan(plan, {"REPIN": "0"})
        self.assertEqual(explicit["REPIN"], "0")

    def test_plan_explains_hot_warm_and_cold_placement(self):
        plan = build_plan(self.model, ram_gb=4, vram_gb=0,
                          available_memory=4 * GB, available_disk=1, gpus=[])
        self.assertEqual([item["target"] for item in plan["decisions"]],
                         ["VRAM", "RAM", "Disk"])
        self.assertIn("quality-preserving yes", format_plan(plan))
        self.assertIn("expected_bottleneck", plan)


if __name__ == "__main__":
    unittest.main()
