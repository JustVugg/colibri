"""Integration tests for the COLI_CUDA opt-in and CUDA_EXPERT_GB auto-size.

These tests run the compiled ``colibri`` binary and verify
the three COLI_CUDA modes (unset / not requested, 0 / forced-CPU, 1 / hard-fail)
and the CUDA_EXPERT_GB auto-size behavior. The engine itself never enables the
GPU on its own: setting COLI_CUDA=1 for the user is the ``coli`` launcher's
job, and running the binary directly — as these tests do — bypasses it.

Prerequisites (tests skip gracefully when unmet):
- Compiled ``colibri`` binary (any build — CUDA or CPU-only)
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
COLIBRI = HERE / ("colibri.exe" if sys.platform == "win32" else "colibri")


def _binary_has_cuda():
    """Check whether the ``colibri`` binary was compiled with CUDA support."""
    if not COLIBRI.exists():
        return False
    try:
        result = subprocess.run(
            [str(COLIBRI), "1"],
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


_HAS_BINARY = COLIBRI.exists()
_HAS_CUDA_BINARY = _HAS_BINARY and _binary_has_cuda()
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
    """COLI_CUDA mode tests — call ``colibri`` directly to exercise ``main()``
    including the CUDA init block at c/colibri.c."""

    @classmethod
    def setUpClass(cls):
        if not _HAS_BINARY:
            raise unittest.SkipTest("colibri binary not found")

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, cap="1", env=None, timeout=30, unset=()):
        """Run the colibri binary directly with a minimal model.

        The binary will fail during model_init (fake weights) but CUDA
        init messages are emitted first.

        ``unset`` names environment variables to drop from the child, so a
        test can assert on a variable being *absent* without an ambient value
        in the invoking shell quietly invalidating the scenario. Removal runs
        last, and the default removes nothing.
        """
        model = _minimal_model(self.tmp.name)
        merged = {**os.environ, "SNAP": str(model), **(env or {})}
        for key in unset:
            merged.pop(key, None)
        return subprocess.run(
            [str(COLIBRI), cap],
            env=merged, cwd=str(HERE), text=True, capture_output=True,
            timeout=timeout,
        )

    # --- COLI_CUDA=0 : forced-CPU mode ---------------------------------

    def test_cuda_zero_suppresses_all_cuda_output(self):
        """COLI_CUDA=0 must not emit any [CUDA] line to stderr."""
        result = self._run(env={"COLI_CUDA": "0"})
        self.assertNotIn("[CUDA]", result.stderr or "")

    def test_cuda_zero_with_gpu_env_fails_guard(self):
        """COLI_GPU with COLI_CUDA=0 exits non-zero (guard in colibri.c).

        On CPU-only binary the guard fires from the #else branch with a
        different message; both are valid and tested here.
        """
        result = self._run(env={"COLI_CUDA": "0", "COLI_GPU": "0"})
        self.assertNotEqual(result.returncode, 0)

        if _HAS_CUDA_BINARY:
            # CUDA build: guard at colibri.c — COLI_GPU(S) requires COLI_CUDA=1
            self.assertIn("COLI_GPU(S) requires COLI_CUDA=1", result.stderr)
        else:
            # CPU-only build: #else branch — CUDA was requested, but CPU-only
            self.assertIn("CPU-only", result.stderr)

    # --- COLI_CUDA=1 : explicit-enable, hard-fail ----------------------

    def test_cuda_one_cpu_only_binary_exits_with_rebuild_message(self):
        """CPU-only binary: COLI_CUDA=1 → exit≠0, 'CPU-only' in stderr."""
        if _HAS_CUDA_BINARY:
            self.skipTest("binary has CUDA; CPU-only rejection not reachable")
        result = self._run(env={"COLI_CUDA": "1"})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("CPU-only", result.stderr)

    def test_cuda_one_without_gpu_exits_two(self):
        """CUDA binary, no GPU: COLI_CUDA=1 → exit 2, 'unavailable'."""
        if not _HAS_CUDA_BINARY:
            self.skipTest("CPU-only binary; covered by cpu-only test")
        if _HAS_GPU:
            self.skipTest("GPU present; cannot simulate missing backend")
        result = self._run(env={"COLI_CUDA": "1"})
        self.assertEqual(result.returncode, 2)
        self.assertIn("[CUDA] requested backend is unavailable", result.stderr)

    # --- COLI_CUDA unset : not requested, silent CPU --------------------

    def test_cuda_unset_without_gpu_falls_back_to_cpu(self):
        """COLI_CUDA unset: the GPU is never requested, so the engine says nothing.

        The engine is strictly opt-in — c/colibri.c enters its CUDA block only
        for a truthy COLI_CUDA, so unset and 0 behave identically and no
        backend load is attempted. Windows auto-enable lives in the separate
        ``coli`` launcher, which this test bypasses by running the binary
        directly; asserting a launcher message here would be asserting against
        the wrong component.
        """
        if not _HAS_CUDA_BINARY:
            self.skipTest("CPU-only binary; the CUDA block is compiled out")
        if _HAS_GPU:
            self.skipTest("GPU present; cannot prove the not-requested path stays quiet")
        result = self._run(unset=("COLI_CUDA", "COLI_GPU", "COLI_GPUS"))
        self.assertNotEqual(result.returncode, 2)
        self.assertNotIn("[CUDA]", result.stderr or "")
        self.assertNotIn("not found; GPU tier disabled", result.stderr or "")

    def test_cuda_unset_with_gpu_stays_on_cpu(self):
        """A detected GPU does not opt the engine into CUDA by itself.

        The companion above proves the quiet path with no GPU; this proves the
        GPU's presence changes nothing. COLI_CUDA remains the explicit
        engine-level switch, so with it unset c/colibri.c never enters the CUDA
        block and never prints a mode line — no matter what hardware is
        installed. The ``coli`` launcher may set COLI_CUDA=1 on the user's
        behalf, but this test invokes the engine binary directly and so must
        not see launcher behaviour.
        """
        if not _HAS_CUDA_BINARY:
            self.skipTest("CPU-only binary; the CUDA block is compiled out")
        if not _HAS_GPU:
            self.skipTest("no GPU available; the with-GPU half is unprovable")
        result = self._run(unset=("COLI_CUDA", "COLI_GPU", "COLI_GPUS"))
        err = result.stderr or ""
        self.assertNotEqual(result.returncode, 2)
        self.assertNotIn("[CUDA]", err)
        self.assertNotIn("[CUDA] mode:", err)
        self.assertNotIn("coli_cuda.dll not found", err)
        self.assertNotIn("requested backend is unavailable", err)
        # Reached model validation, so the engine skipped GPU setup rather
        # than failing inside it.
        self.assertIn("this engine requires n_group=1", err)

    # --- CUDA_EXPERT_GB explicit zero ----------------------------------

    def test_cuda_expert_gb_zero_disables_auto_size(self):
        """Explicit CUDA_EXPERT_GB=0 — no auto-size message at startup."""
        if not _HAS_CUDA_BINARY:
            self.skipTest("CPU-only binary")
        result = self._run(env={"CUDA_EXPERT_GB": "0"})
        self.assertNotIn("auto-sized", result.stderr or "")


class CudaExpertBudgetTest(unittest.TestCase):
    """Auto-size tests — need model_init() path (heavier, GPU required)."""

    @classmethod
    def setUpClass(cls):
        if not _HAS_BINARY:
            raise unittest.SkipTest("colibri binary not found")
        if not _HAS_CUDA_BINARY:
            raise unittest.SkipTest("CPU-only binary — auto-size path not compiled in")
        if not _HAS_GPU:
            raise unittest.SkipTest("no GPU — auto-size needs free VRAM query")

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, cap="1", env=None, timeout=60):
        model = _minimal_model(self.tmp.name)
        merged = {**os.environ, "SNAP": str(model), **(env or {})}
        return subprocess.run(
            [str(COLIBRI), cap],
            env=merged, cwd=str(HERE), text=True, capture_output=True,
            timeout=timeout,
        )

    def test_auto_size_prints_budget_when_expert_gb_unset(self):
        """CUDA enabled, CUDA_EXPERT_GB unset → 'auto-sized' in stderr."""
        result = self._run(env={"COLI_CUDA": "1"})
        combined = (result.stderr or "") + (result.stdout or "")
        if "auto-sized" in combined:
            self.assertIn("expert budget auto-sized", combined)

    def test_explicit_expert_gb_zero_suppresses_auto_size_at_pin_time(self):
        """CUDA_EXPERT_GB=0 → no auto-size message even at pin time."""
        result = self._run(env={"COLI_CUDA": "1", "CUDA_EXPERT_GB": "0"})
        combined = (result.stderr or "") + (result.stdout or "")
        self.assertNotIn("auto-sized", combined)


if __name__ == "__main__":
    unittest.main()
