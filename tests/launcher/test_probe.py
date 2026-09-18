"""Compatibility checks for the installed Colibri capability bridge."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from colibri.launcher.probe import _cuda_support


class LegacyCudaProbeTests(unittest.TestCase):
    def test_legacy_cuda_requires_selected_engine_and_cli_launch_gate(self):
        # Colibri 1.10.1's no-argument cuda_binary only inspects its GLM engine.
        # env_for_engine nevertheless requires that predicate for --gpu/--vram,
        # even when a different selected engine has its own CUDA linkage.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            selected = root / "selected.exe"
            selected.write_bytes(b"[CUDA] mode: routed experts\0coli_cuda.dll")
            (root / "coli_cuda.dll").write_bytes(b"installed runtime")
            for default_supported, expected in ((False, False), (True, True)):
                def legacy_probe():
                    return default_supported

                with self.subTest(default_supported=default_supported), patch(
                    "colibri.launcher.probe.sys.platform", "win32"
                ):
                    supported, reason = _cuda_support(
                        {"cuda_binary": legacy_probe}, selected, "glm53"
                    )
                    self.assertEqual(supported, expected, reason)
                    if not expected:
                        self.assertIn("Update Colibri", reason)

    def test_legacy_probe_does_not_promote_cpu_hip_or_missing_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            selected = root / "selected.exe"
            runtime = root / "coli_cuda.dll"
            for label, image, runtime_present, explanation in (
                ("CPU engine beside CUDA runtime", b"CPU engine", True, "no verified NVIDIA CUDA support"),
                ("HIP engine", b"[CUDA] mode: routed experts\0coli_hip.dll", True, "AMD HIP"),
                ("missing runtime", b"[CUDA] mode: routed experts\0coli_cuda.dll", False, "coli_cuda.dll"),
            ):
                selected.write_bytes(image)
                if runtime_present:
                    runtime.write_bytes(b"installed runtime")
                else:
                    runtime.unlink(missing_ok=True)
                with self.subTest(label=label), patch(
                    "colibri.launcher.probe.sys.platform", "win32"
                ):
                    for default_supported in (False, True):
                        supported, reason = _cuda_support(
                            {"cuda_binary": lambda: default_supported}, selected, "glm53"
                        )
                        self.assertFalse(supported)
                        self.assertNotIn("Update Colibri", reason)
                        self.assertIn(explanation, reason)
