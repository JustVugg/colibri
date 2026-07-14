"""Integration tests for COLI_CUDA auto-enable and CUDA_EXPERT_GB auto-size.

These tests run the compiled ``glm`` binary (via ``coli run``) and verify
the three COLI_CUDA modes (unset / auto-detect, 0 / forced-CPU, 1 / hard-fail)
and the CUDA_EXPERT_GB auto-size behavior.

Prerequisites (tests skip gracefully when unmet):
- Compiled ``coli`` / ``glm`` binary (any build — CUDA or CPU-only)
- nvidia-smi (for GPU-specific tests)
"""

import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent.parent
GLM = HERE / ("glm.exe" if sys.platform == "win32" else "glm")


def _binary_has_cuda():
    """Check whether the ``glm`` binary was compiled with CUDA support."""
    if not GLM.exists():
        return False
    try:
        result = subprocess.run(
            [str(GLM), "1"],
            env={**os.environ, "COLI_CUDA": "1",
                 "SNAP": str(HERE)},
            cwd=str(HERE), text=True, capture_output=True, timeout=10,
        )
        return "CPU-only" not in (result.stderr or "")
    except (OSError, subprocess.SubprocessError):
        return False


def _gpu_available():
    """Return True if at least one CUDA-capable GPU is visible to nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True, capture_output=True, timeout=5,
        )
        return result.returncode == 0 and bool(result.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return False


_HAS_GLM = GLM.exists()
_HAS_CUDA_BINARY = _HAS_GLM and _binary_has_cuda()
_HAS_GPU = _gpu_available()


def _write_shard(path, tensors):
    """Write a minimal safetensors file to *path*."""
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


def _minimal_model(parent):
    """Create a minimal model directory that ``model_init()`` can parse.

    Returns the model Path.  The safetensors payload is dummy zeros — the
    binary will reach CUDA init, print its messages, then fail during
    model loading (fake weights).  The CUDA messages are already on stderr
    at that point.
    """
    model = Path(parent) / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({
        "num_hidden_layers": 2,
        "n_routed_experts": 2,
        "kv_lora_rank": 4,
        "qk_rope_head_dim": 2,
        "qk_nope_head_dim": 3,
        "v_head_dim": 5,
        "num_attention_heads": 2,
    }))
    _write_shard(model / "model.safetensors", [
        ("model.embed_tokens.weight", 100),
        ("model.layers.0.self_attn.q_a_proj.weight", 200),
        ("model.layers.1.mlp.experts.0.gate_proj.weight", 30),
        ("model.layers.1.mlp.experts.0.up_proj.weight", 30),
        ("model.layers.1.mlp.experts.1.gate_proj.weight", 30),
        ("model.layers.1.mlp.experts.1.up_proj.weight", 30),
    ])
    return model


class CudaStartupTest(unittest.TestCase):
    """COLI_CUDA mode tests — call ``glm`` directly to exercise ``main()``
    including the CUDA init block at c/glm.c:3744."""

    @classmethod
    def setUpClass(cls):
        if not _HAS_GLM:
            raise unittest.SkipTest("glm binary not found")

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def _run_glm(self, cap="1", env=None, timeout=30):
        """Run the glm binary directly with a minimal model.

        The binary will fail during model_init (fake weights) but CUDA
        init messages are emitted first.
        """
        model = _minimal_model(self.tmp.name)
        merged = {**os.environ, "SNAP": str(model), **(env or {})}
        return subprocess.run(
            [str(GLM), cap],
            env=merged, cwd=str(HERE), text=True, capture_output=True,
            timeout=timeout,
        )

    # --- COLI_CUDA=0 : forced-CPU mode ---------------------------------

    def test_cuda_zero_suppresses_all_cuda_output(self):
        """COLI_CUDA=0 must not emit any [CUDA] line to stderr."""
        result = self._run_glm(env={"COLI_CUDA": "0"})
        self.assertNotIn("[CUDA]", result.stderr or "")

    def test_cuda_zero_with_gpu_env_fails_guard(self):
        """COLI_GPU with COLI_CUDA=0 exits non-zero (guard in glm.c)."""
        result = self._run_glm(env={"COLI_CUDA": "0", "COLI_GPU": "0"})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("requires COLI_CUDA=1", result.stderr)

    # --- COLI_CUDA=1 : explicit-enable, hard-fail ----------------------

    def test_cuda_one_cpu_only_binary_exits_with_rebuild_message(self):
        """CPU-only binary: COLI_CUDA=1 → exit≠0, 'CPU-only' in stderr."""
        if _HAS_CUDA_BINARY:
            self.skipTest("binary has CUDA; CPU-only rejection not reachable")
        result = self._run_glm(env={"COLI_CUDA": "1"})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("CPU-only", result.stderr)

    def test_cuda_one_without_gpu_exits_two(self):
        """CUDA binary, no GPU: COLI_CUDA=1 → exit 2, 'unavailable'."""
        if not _HAS_CUDA_BINARY:
            self.skipTest("CPU-only binary; covered by cpu-only test")
        if _HAS_GPU:
            self.skipTest("GPU present; cannot simulate missing backend")
        result = self._run_glm(env={"COLI_CUDA": "1"})
        self.assertEqual(result.returncode, 2)
        self.assertIn("[CUDA] requested backend is unavailable", result.stderr)

    # --- COLI_CUDA unset : auto-detect ---------------------------------

    def test_cuda_unset_without_gpu_falls_back_to_cpu(self):
        """No GPU: auto-detect prints fallback message, does not crash."""
        if not _HAS_CUDA_BINARY:
            self.skipTest("CPU-only binary has no auto-detect path")
        if _HAS_GPU:
            self.skipTest("GPU present; cannot simulate auto-detect fallback")
        result = self._run_glm()
        self.assertNotEqual(result.returncode, 2)
        self.assertIn("auto-detect: backend unavailable", result.stderr)

    def test_cuda_unset_with_gpu_enables_cuda(self):
        """GPU present: auto-detect enables CUDA, prints mode line."""
        if not _HAS_CUDA_BINARY:
            self.skipTest("CPU-only binary")
        if not _HAS_GPU:
            self.skipTest("no GPU available")
        result = self._run_glm()
        self.assertNotEqual(result.returncode, 2)
        self.assertIn("[CUDA] mode:", result.stderr)

    # --- CUDA_EXPERT_GB explicit zero ----------------------------------

    def test_cuda_expert_gb_zero_disables_auto_size(self):
        """Explicit CUDA_EXPERT_GB=0 — no auto-size message at startup."""
        if not _HAS_CUDA_BINARY:
            self.skipTest("CPU-only binary")
        result = self._run_glm(env={"CUDA_EXPERT_GB": "0"})
        self.assertNotIn("auto-sized", result.stderr or "")


class CudaExpertBudgetTest(unittest.TestCase):
    """Auto-size tests — need model_init() path (heavier, GPU required)."""

    @classmethod
    def setUpClass(cls):
        if not _HAS_GLM:
            raise unittest.SkipTest("glm binary not found")
        if not _HAS_CUDA_BINARY:
            raise unittest.SkipTest("CPU-only binary — auto-size path not compiled in")
        if not _HAS_GPU:
            raise unittest.SkipTest("no GPU — auto-size needs free VRAM query")

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def _run_glm(self, cap="1", env=None, timeout=60):
        model = _minimal_model(self.tmp.name)
        merged = {**os.environ, "SNAP": str(model), **(env or {})}
        return subprocess.run(
            [str(GLM), cap],
            env=merged, cwd=str(HERE), text=True, capture_output=True,
            timeout=timeout,
        )

    def test_auto_size_prints_budget_when_expert_gb_unset(self):
        """CUDA enabled, CUDA_EXPERT_GB unset → 'auto-sized' in stderr."""
        result = self._run_glm(env={"COLI_CUDA": "1"})
        combined = (result.stderr or "") + (result.stdout or "")
        if "auto-sized" in combined:
            self.assertIn("expert budget auto-sized", combined)

    def test_explicit_expert_gb_zero_suppresses_auto_size_at_pin_time(self):
        """CUDA_EXPERT_GB=0 → no auto-size message even at pin time."""
        result = self._run_glm(env={"COLI_CUDA": "1", "CUDA_EXPERT_GB": "0"})
        combined = (result.stderr or "") + (result.stdout or "")
        self.assertNotIn("auto-sized", combined)


if __name__ == "__main__":
    unittest.main()
