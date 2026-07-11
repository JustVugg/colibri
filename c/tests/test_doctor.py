import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from doctor import exit_code, format_doctor, run_doctor
from resource_plan import GB


def write_shard(path, tensors):
    offset = 0
    header = {}
    payload = b""
    for name, size in tensors:
        header[name] = {"dtype": "U8", "shape": [size],
                        "data_offsets": [offset, offset + size]}
        payload += b"\0" * size
        offset += size
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)


class DoctorTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.model = self.root / "model"
        self.model.mkdir()
        (self.model / "config.json").write_text(json.dumps({
            "num_hidden_layers": 2,
            "n_routed_experts": 2,
            "kv_lora_rank": 4,
            "qk_rope_head_dim": 2,
            "qk_nope_head_dim": 3,
            "v_head_dim": 5,
            "num_attention_heads": 2,
        }))
        (self.model / "tokenizer.json").write_text("{}")
        write_shard(self.model / "model.safetensors", [
            ("model.embed_tokens.weight", 100),
            ("model.layers.0.self_attn.q_a_proj.weight", 200),
            ("model.layers.1.mlp.experts.0.gate_proj.weight", 30),
            ("model.layers.1.mlp.experts.0.up_proj.weight", 30),
            ("model.layers.1.mlp.experts.1.gate_proj.weight", 30),
            ("model.layers.1.mlp.experts.1.up_proj.weight", 30),
        ])
        self.engine = self.root / "glm"
        self.engine.write_text("#!/bin/sh\nexit 0\n")
        self.engine.chmod(0o755)

    def tearDown(self):
        self.tmp.cleanup()

    def report(self, **overrides):
        arguments = {
            "model": self.model,
            "ram_gb": 16,
            "context": 32,
            "gpu_indices": [],
            "vram_gb": 0,
            "engine_path": self.engine,
            "available_memory": 32 * GB,
            "available_disk": 100 * GB,
            "gpus": [],
            "linkage": {"linked": False, "missing": False},
        }
        arguments.update(overrides)
        return run_doctor(**arguments)

    @staticmethod
    def checks_by_id(report):
        return {check["id"]: check for check in report["checks"]}

    def test_healthy_cpu_install_has_versioned_report(self):
        report = self.report()
        checks = self.checks_by_id(report)

        self.assertEqual(report["schema_version"], 1)
        self.assertEqual(report["status"], "ok")
        self.assertIsNotNone(report["plan"])
        self.assertEqual(checks["accelerator.cuda"]["status"], "skip")
        self.assertEqual(checks["memory.ram"]["status"], "pass")
        self.assertEqual(checks["model.shards"]["details"]["shards"], 1)
        self.assertEqual(exit_code(report), 0)

    def test_missing_model_collects_failures_instead_of_stopping_early(self):
        report = self.report(model=self.root / "missing")
        checks = self.checks_by_id(report)

        self.assertEqual(report["status"], "error")
        self.assertEqual(checks["model.path"]["status"], "fail")
        self.assertEqual(checks["model.config"]["status"], "fail")
        self.assertEqual(checks["model.tokenizer"]["status"], "fail")
        self.assertEqual(checks["model.shards"]["status"], "fail")
        self.assertEqual(checks["storage.disk"]["status"], "skip")
        self.assertIsNone(report["plan"])
        self.assertEqual(exit_code(report), 1)

    def test_non_executable_engine_and_excessive_ram_budget_fail(self):
        self.engine.chmod(0o644)
        report = self.report(ram_gb=40)
        checks = self.checks_by_id(report)

        self.assertEqual(checks["engine.binary"]["status"], "fail")
        self.assertEqual(checks["memory.ram"]["status"], "fail")
        self.assertEqual(report["status"], "error")

    def test_requested_missing_gpu_is_a_failure(self):
        report = self.report(gpu_indices=[1])
        check = self.checks_by_id(report)["accelerator.cuda"]

        self.assertEqual(check["status"], "fail")
        self.assertEqual(check["details"], {"requested": [1], "detected": []})
        self.assertEqual(exit_code(report), 1)

    def test_cpu_engine_with_detected_gpu_is_only_a_warning(self):
        gpu = {"index": 0, "name": "fixture", "total_bytes": 12 * GB,
               "free_bytes": 10 * GB}
        report = self.report(gpu_indices=None, gpus=[gpu])
        check = self.checks_by_id(report)["accelerator.cuda"]

        self.assertEqual(check["status"], "warn")
        self.assertEqual(report["status"], "warning")
        self.assertEqual(exit_code(report), 0)

    def test_missing_cuda_runtime_is_a_failure(self):
        gpu = {"index": 0, "name": "fixture", "total_bytes": 12 * GB,
               "free_bytes": 10 * GB}
        report = self.report(gpu_indices=[0], gpus=[gpu],
                             linkage={"linked": False, "missing": True})

        self.assertEqual(
            self.checks_by_id(report)["accelerator.cuda"]["summary"],
            "CUDA runtime library is missing",
        )
        self.assertEqual(report["status"], "error")

    def test_text_format_contains_checks_plan_and_result(self):
        output = format_doctor(self.report())

        self.assertIn("model.path", output)
        self.assertIn("disk   backing store", output)
        self.assertTrue(output.endswith("result ok"))

    def test_cli_json_is_machine_readable_without_loading_model(self):
        cli = Path(__file__).parents[1] / "coli"
        run = subprocess.run([
            sys.executable, str(cli), "doctor", "--model", str(self.model),
            "--gpu", "none", "--ram", "16", "--ctx", "32", "--json",
        ], text=True, capture_output=True, check=False)

        # The repository engine may be absent; doctor must still return one complete JSON report.
        self.assertIn(run.returncode, (0, 1))
        report = json.loads(run.stdout)
        self.assertEqual(report["schema_version"], 1)
        self.assertEqual(Path(report["model"]), self.model.resolve())
        self.assertIn(report["status"], ("ok", "warning", "error"))
        self.assertNotIn("\033", run.stdout)
        self.assertTrue(run.stdout.lstrip().startswith("{"))
        self.assertTrue(run.stdout.rstrip().endswith("}"))


if __name__ == "__main__":
    unittest.main()
