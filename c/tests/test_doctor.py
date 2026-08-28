import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from doctor import (
    _tensor_layout,
    cuda_linkage,
    exit_code,
    format_doctor,
    run_doctor,
)
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
            "model_type": "glm_moe_dsa",
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
            ("model.norm.weight", 8),
            ("lm_head.weight", 100),
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
        self.assertEqual(report["mode"], "standard")
        self.assertEqual(report["status"], "ok")
        self.assertIsNotNone(report["plan"])
        self.assertEqual(checks["accelerator.gpu"]["status"], "skip")
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

        # On Windows chmod(0o644) does not remove executability (NTFS has no
        # execute bit; os.access(X_OK) is always True for existing files), so
        # the engine.binary check stays "pass" there. The excessive-RAM check
        # (memory.ram) is platform-independent and must still fail. (#141)
        if sys.platform == "win32":
            self.assertEqual(checks["engine.binary"]["status"], "pass")
        else:
            self.assertEqual(checks["engine.binary"]["status"], "fail")
        self.assertEqual(checks["memory.ram"]["status"], "fail")
        self.assertEqual(report["status"], "error")

    def test_requested_missing_gpu_is_a_failure(self):
        report = self.report(gpu_indices=[1])
        check = self.checks_by_id(report)["accelerator.gpu"]

        self.assertEqual(check["status"], "fail")
        self.assertEqual(check["details"], {"requested": [1], "detected": []})
        self.assertEqual(exit_code(report), 1)

    def test_cpu_engine_with_detected_gpu_is_only_a_warning(self):
        gpu = {"index": 0, "name": "fixture", "total_bytes": 12 * GB,
               "free_bytes": 10 * GB}
        report = self.report(gpu_indices=None, gpus=[gpu])
        check = self.checks_by_id(report)["accelerator.gpu"]

        self.assertEqual(check["status"], "warn")
        self.assertEqual(report["status"], "warning")
        self.assertEqual(exit_code(report), 0)

    def test_gpu_check_uses_backend_neutral_identifier(self):
        gpu = {"index": 0, "name": "Intel Arc B570", "total_bytes": 10 * GB,
               "free_bytes": 9 * GB}
        report = self.report(gpu_indices=None, gpus=[gpu])
        checks = self.checks_by_id(report)
        self.assertIn("accelerator.gpu", checks)
        self.assertNotIn("accelerator.cuda", checks)

    def test_missing_cuda_runtime_is_a_failure(self):
        gpu = {"index": 0, "name": "fixture", "total_bytes": 12 * GB,
               "free_bytes": 10 * GB}
        report = self.report(gpu_indices=[0], gpus=[gpu],
                             linkage={"linked": False, "missing": True})

        self.assertEqual(
            self.checks_by_id(report)["accelerator.gpu"]["summary"],
            "GPU runtime library is missing",
        )
        self.assertEqual(report["status"], "error")

    # #379: doctor surfaces the cached F_NOCACHE probe read-only (S4) -- it
    # never re-measures storage itself, only reflects what colibri.c already wrote.
    # --- Windows backend-artifact linkage ---------------------------------
    # c/backend_loader.c compiles exactly one backend basename into the host:
    # COLI_BACKEND_DLL is "coli_hip.dll" under COLI_HIP_DLL and "coli_cuda.dll"
    # otherwise. So the binary states which artifact it will LoadLibrary, and
    # doctor can check for that one instead of assuming CUDA. This validates the
    # host/artifact contract only -- it says nothing about the runtime binding
    # or about any GPU actually computing.

    def _win_engine(self, backend_dll, artifacts=()):
        engine = self.root / "colibri.exe"
        engine.write_bytes(b"...[CUDA] mode: routed experts..."
                           + backend_dll.encode() + b"...")
        for artifact in artifacts:
            (self.root / artifact).write_bytes(b"")
        return engine

    def _win_linkage(self, engine):
        # Only sys.platform is faked. Faking os.name as well would make
        # pathlib build a WindowsPath from the POSIX fixture path, which does
        # not resolve on Linux/macOS, so cuda_linkage would bail at its
        # is_file() guard before reaching the backend-marker logic and every
        # assertion below would compare against a false negative.
        with mock.patch.object(sys, "platform", "win32"):
            return cuda_linkage(engine)

    def test_windows_cuda_host_accepts_its_own_backend(self):
        engine = self._win_engine("coli_cuda.dll", ["coli_cuda.dll"])
        self.assertEqual(self._win_linkage(engine),
                         {"linked": True, "missing": False})

    def test_windows_hip_host_accepts_its_own_backend(self):
        engine = self._win_engine("coli_hip.dll", ["coli_hip.dll"])
        self.assertEqual(self._win_linkage(engine),
                         {"linked": True, "missing": False})

    def test_windows_hip_host_is_not_satisfied_by_the_cuda_backend(self):
        engine = self._win_engine("coli_hip.dll", ["coli_cuda.dll"])
        self.assertEqual(self._win_linkage(engine),
                         {"linked": False, "missing": True})

    def test_windows_cuda_host_is_not_satisfied_by_the_hip_backend(self):
        engine = self._win_engine("coli_cuda.dll", ["coli_hip.dll"])
        self.assertEqual(self._win_linkage(engine),
                         {"linked": False, "missing": True})

    def test_windows_missing_backend_artifact_still_fails(self):
        engine = self._win_engine("coli_hip.dll")
        self.assertEqual(self._win_linkage(engine),
                         {"linked": False, "missing": True})

    def test_windows_cpu_only_engine_is_not_a_gpu_build(self):
        engine = self.root / "colibri.exe"
        engine.write_bytes(b"a plain CPU build with no backend loader")
        self.assertEqual(self._win_linkage(engine),
                         {"linked": False, "missing": False})

    def test_hip_host_with_its_backend_reports_gpu_available(self):
        # End-to-end: the same host that previously reported a hard error
        # ("GPU runtime library is missing") now passes, with #903's
        # backend-neutral wording preserved.
        engine = self._win_engine("coli_hip.dll", ["coli_hip.dll"])
        report = self.report(gpu_indices=None, engine_path=engine,
                             gpus=[{"index": 0, "name": "AMD Radeon(TM) 8060S Graphics",
                                    "total_bytes": 78 * GB, "free_bytes": None,
                                    "unified_memory": True}],
                             linkage=self._win_linkage(engine))
        check = self.checks_by_id(report)["accelerator.gpu"]
        self.assertEqual(check["status"], "pass")
        self.assertNotIn("CUDA", check["summary"])
        self.assertNotIn("NVIDIA", check["summary"])

    def test_ssd_probe_check_skips_when_not_yet_cached(self):
        checks = self.checks_by_id(self.report())
        self.assertEqual(checks["storage.ssd_probe"]["status"], "skip")

    def test_ssd_probe_check_passes_and_reports_cached_value(self):
        # byte-exact (write_bytes): Windows text mode would CRLF-translate and the
        # strict reader would (correctly) reject the fixture as garbage
        (self.model / ".coli_ssd").write_bytes(f"v2 14.3 {os.stat(self.model).st_dev}\n".encode("ascii"))
        checks = self.checks_by_id(self.report())
        self.assertEqual(checks["storage.ssd_probe"]["status"], "pass")
        self.assertEqual(checks["storage.ssd_probe"]["details"]["gbs"], 14.3)

    def test_ssd_probe_wording_names_why_a_cache_is_pending(self):
        # #386 r2, F10: "no cached probe yet" is a lie when a file exists --
        # each untrusted state names what will actually happen instead.
        cases = (  # byte-exact fixtures: no text-mode CRLF on Windows
            (b"14.3\n", "legacy cache pending engine upgrade"),
            (f"v2 14.3 {os.stat(self.model).st_dev + 1}\n".encode("ascii"), "cache from another volume"),
            (b"not-a-number\n", "unreadable cache"),
        )
        for content, expected in cases:
            (self.model / ".coli_ssd").write_bytes(content)
            check = self.checks_by_id(self.report())["storage.ssd_probe"]
            self.assertEqual(check["status"], "skip", content)
            self.assertIn(expected, check["summary"], content)
            self.assertNotIn("no cached probe yet", check["summary"], content)
        (self.model / ".coli_ssd").unlink()
        check = self.checks_by_id(self.report())["storage.ssd_probe"]
        self.assertIn("no cached probe yet", check["summary"])

    def test_ssd_probe_check_never_emits_json_infinity(self):
        # float("inf") used to sail through the old reader; json.dumps renders
        # it as the bare literal Infinity, which is not JSON -- machine
        # consumers of `coli doctor --json` would then fail to parse the whole
        # report. The strict v2 grammar bans inf/nan outright (#386 fix round).
        (self.model / ".coli_ssd").write_bytes(b"inf\n")
        report = self.report()
        checks = self.checks_by_id(report)
        self.assertEqual(checks["storage.ssd_probe"]["status"], "skip")
        encoded = json.dumps(report, indent=2, allow_nan=False)  # raises on inf/nan
        json.loads(encoded)

    def test_text_format_contains_checks_plan_and_result(self):
        output = format_doctor(self.report())

        self.assertIn("model.path", output)
        self.assertIn("disk   0.0 GB cold experts", output)
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

    def test_deep_check_validates_every_tensor_layout(self):
        report = self.report(deep=True)
        checks = self.checks_by_id(report)

        self.assertEqual(report["mode"], "deep")
        self.assertEqual(checks["model.container"]["status"], "pass")
        self.assertEqual(checks["model.container"]["details"]["shards"], 1)
        self.assertEqual(checks["model.container"]["details"]["tensors"], 8)
        self.assertFalse(checks["model.container"]["details"]["payload_hashing"])
        self.assertEqual(checks["model.shard_sequence"]["status"], "skip")
        self.assertEqual(checks["model.required"]["status"], "pass")
        self.assertEqual(checks["model.index"]["status"], "skip")
        self.assertEqual(checks["storage.mirror"]["status"], "skip")

    def test_tensor_layout_accepts_engine_supported_dtypes(self):
        # I64 (Hash-MoE tid2eid) and F8 dtypes are read by the engine (st.h
        # st_dtype_code) but were absent from the validator's table; --deep must
        # accept them at the byte sizes st_dtype_esz reports (I64/U64 -> 8, F8 -> 1).
        cases = {
            "I64": (8, 4),
            "U64": (8, 4),
            "F8_E4M3": (1, 4),
            "F8_E4M3FN": (1, 4),
            "float8_e4m3fn": (1, 4),
            "F8_E8M0": (1, 4),
            "F8_E8M0FNU": (1, 4),
        }
        for dtype, (size, elements) in cases.items():
            span = size * elements
            meta = {"dtype": dtype, "shape": [elements], "data_offsets": [0, span]}
            self.assertEqual(_tensor_layout(meta, span), (0, span))
            bad = {"dtype": dtype, "shape": [elements], "data_offsets": [0, span - 1]}
            with self.assertRaises(ValueError):
                _tensor_layout(bad, span)

    def test_deep_check_rejects_overlapping_tensor_ranges(self):
        header = {
            "first": {"dtype": "U8", "shape": [4], "data_offsets": [0, 4]},
            "second": {"dtype": "U8", "shape": [4], "data_offsets": [2, 6]},
        }
        raw = json.dumps(header).encode()
        shard = self.model / "model.safetensors"
        shard.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0" * 6)

        checks = self.checks_by_id(self.report(deep=True))

        self.assertEqual(checks["model.container"]["status"], "fail")
        self.assertIn("overlap", checks["model.container"]["summary"])

    def test_deep_check_rejects_duplicate_tensor_names_across_shards(self):
        write_shard(self.model / "second.safetensors", [
            ("model.embed_tokens.weight", 1),
        ])

        checks = self.checks_by_id(self.report(deep=True))

        self.assertEqual(checks["model.container"]["status"], "fail")
        self.assertIn("duplicate tensor", checks["model.container"]["summary"])

    def test_deep_check_rejects_gap_in_converter_shard_sequence(self):
        write_shard(self.model / "out-00000.safetensors", [("zero.weight", 1)])
        write_shard(self.model / "out-00002.safetensors", [("two.weight", 1)])

        checks = self.checks_by_id(self.report(deep=True))

        self.assertEqual(checks["model.container"]["status"], "pass")
        self.assertEqual(checks["model.shard_sequence"]["status"], "fail")
        self.assertEqual(checks["model.shard_sequence"]["details"]["missing_shards"], 1)

    def test_deep_check_rejects_converter_sequence_not_starting_at_zero(self):
        write_shard(self.model / "out-00005.safetensors", [("five.weight", 1)])
        write_shard(self.model / "out-00006.safetensors", [("six.weight", 1)])

        checks = self.checks_by_id(self.report(deep=True))

        self.assertEqual(checks["model.container"]["status"], "pass")
        self.assertEqual(checks["model.shard_sequence"]["status"], "fail")
        self.assertEqual(checks["model.shard_sequence"]["details"]["first_shard"], 5)
        self.assertEqual(checks["model.shard_sequence"]["details"]["missing_shards"], 5)

    def test_deep_check_handles_sparse_large_converter_index(self):
        write_shard(self.model / "out-1000000000000.safetensors", [
            ("sparse.weight", 1),
        ])

        checks = self.checks_by_id(self.report(deep=True))
        sequence = checks["model.shard_sequence"]

        self.assertEqual(sequence["status"], "fail")
        self.assertEqual(sequence["details"]["last_shard"], 1_000_000_000_000)
        self.assertEqual(sequence["details"]["missing_shards"], 1_000_000_000_000)

    def test_deep_check_rejects_mixed_shard_filename_schemes(self):
        write_shard(self.model / "model-00001-of-00001.safetensors", [
            ("hf.weight", 1),
        ])
        write_shard(self.model / "out-00000.safetensors", [("out.weight", 1)])

        checks = self.checks_by_id(self.report(deep=True))

        self.assertEqual(checks["model.container"]["status"], "pass")
        self.assertEqual(checks["model.shard_sequence"]["status"], "fail")
        self.assertIn("mixes", checks["model.shard_sequence"]["summary"])

    def test_deep_check_rejects_missing_core_tensor(self):
        write_shard(self.model / "model.safetensors", [
            ("model.embed_tokens.weight", 1),
        ])

        checks = self.checks_by_id(self.report(deep=True))

        self.assertEqual(checks["model.container"]["status"], "pass")
        self.assertEqual(checks["model.required"]["status"], "fail")
        self.assertEqual(checks["model.required"]["details"]["missing_tensors"], [
            "model.norm.weight",
            "lm_head.weight",
        ])

    def test_deep_check_reports_runtime_equivalent_partial_mirror(self):
        mirror = self.root / "mirror"
        mirror.mkdir()
        primary = self.model / "model.safetensors"
        (mirror / primary.name).write_bytes(primary.read_bytes())

        checks = self.checks_by_id(self.report(deep=True, mirror_dir=mirror))
        mirror_check = checks["storage.mirror"]

        self.assertEqual(mirror_check["status"], "pass")
        self.assertEqual(mirror_check["details"]["accepted_shards"], 1)
        self.assertEqual(mirror_check["details"]["divergent_shards"], 0)
        self.assertTrue(mirror_check["details"]["partial_mirror_allowed"])

    def test_deep_check_warns_when_mirror_header_diverges(self):
        mirror = self.root / "mirror"
        mirror.mkdir()
        write_shard(mirror / "model.safetensors", [("different.weight", 620)])

        checks = self.checks_by_id(self.report(deep=True, mirror_dir=mirror))
        mirror_check = checks["storage.mirror"]

        self.assertEqual(mirror_check["status"], "warn")
        self.assertEqual(mirror_check["details"]["accepted_shards"], 0)
        self.assertEqual(mirror_check["details"]["divergent_shards"], 1)

    def test_deep_check_validates_model_index(self):
        tensors = [
            "model.embed_tokens.weight",
            "model.norm.weight",
            "lm_head.weight",
            "model.layers.0.self_attn.q_a_proj.weight",
            "model.layers.1.mlp.experts.0.gate_proj.weight",
            "model.layers.1.mlp.experts.0.up_proj.weight",
            "model.layers.1.mlp.experts.1.gate_proj.weight",
            "model.layers.1.mlp.experts.1.up_proj.weight",
        ]
        (self.model / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {name: "model.safetensors" for name in tensors},
        }))

        checks = self.checks_by_id(self.report(deep=True))

        self.assertEqual(checks["model.index"]["status"], "pass")
        self.assertEqual(checks["model.index"]["details"]["indexed_tensors"], len(tensors))

    def test_deep_check_reports_non_object_model_index(self):
        (self.model / "model.safetensors.index.json").write_text("[]")

        checks = self.checks_by_id(self.report(deep=True))

        self.assertEqual(checks["model.container"]["status"], "pass")
        self.assertEqual(checks["model.index"]["status"], "fail")
        self.assertIn("not a JSON object", checks["model.index"]["summary"])

    def test_deep_check_bounds_model_index_read(self):
        (self.model / "model.safetensors.index.json").write_text("{}")

        with mock.patch("doctor.MODEL_INDEX_MAX_BYTES", 1):
            checks = self.checks_by_id(self.report(deep=True))

        self.assertEqual(checks["model.container"]["status"], "pass")
        self.assertEqual(checks["model.index"]["status"], "fail")
        self.assertIn("exceeds 1 bytes", checks["model.index"]["summary"])

    def test_cli_deep_json_is_machine_readable(self):
        cli = Path(__file__).parents[1] / "coli"
        run = subprocess.run([
            sys.executable, str(cli), "doctor", "--model", str(self.model),
            "--gpu", "none", "--ram", "16", "--ctx", "32", "--deep", "--json",
        ], text=True, capture_output=True, check=False)

        report = json.loads(run.stdout)
        checks = self.checks_by_id(report)
        self.assertEqual(report["mode"], "deep")
        self.assertIn("model.container", checks)


if __name__ == "__main__":
    unittest.main()
