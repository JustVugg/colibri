"""Apple GPU detection and the explicit Metal setup path."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import setup_hw
import setup_flow
from family_registry import family_by_id


class MetalSetup(unittest.TestCase):
    def test_macos_gpu(self):
        data = {"SPDisplaysDataType": [{"sppci_model": "Apple M5 Pro",
                                       "spdisplays_metal": "spdisplays_supported"}]}
        with patch.object(setup_hw, "host_os", return_value="darwin"), patch.object(
                setup_hw, "_run", return_value=json.dumps(data)):
            self.assertEqual(setup_hw.detect_metal()["name"], "Apple M5 Pro")

    def test_unavailable_gpu(self):
        with patch.object(setup_hw, "host_os", return_value="darwin"), patch.object(
                setup_hw, "_run", return_value="invalid"):
            self.assertIsNone(setup_hw.detect_metal())

    def test_non_macos_does_not_probe(self):
        with patch.object(setup_hw, "host_os", return_value="linux"), patch.object(
                setup_hw, "_run") as probe:
            self.assertIsNone(setup_hw.detect_metal())
            probe.assert_not_called()

    def test_explicit_metal(self):
        hw = {"gpu": {"metal": {"name": "Apple M5 Pro"}}}
        with patch.object(setup_flow, "host_os", return_value="darwin"):
            for name in setup_flow.METAL_ENGINES:
                family = family_by_id(name)
                decision = setup_flow.choose_backend(hw, family, {"can_build": True}, "metal")
                self.assertEqual(decision["backend"], "metal")
                command, _, _ = setup_flow.make_command(family, "metal", {})
                self.assertIn("METAL=1", command)
                env = setup_flow.run_environment("metal", "/tmp/engine", family, hw)
                self.assertEqual(env, {"K3_METAL" if name == "kimi" else "COLI_METAL": "1"})
            with self.assertRaises(setup_flow.SetupError):
                setup_flow.choose_backend(hw, family_by_id("qwen36"), {"can_build": True}, "metal")
            self.assertEqual(setup_flow.choose_backend(hw, family, {}, "auto")["backend"], "cpu")
            self.assertEqual(setup_flow.choose_backend(hw, family, {}, "metal", exclude=("metal",))["backend"], "cpu")

    def test_metal_binary(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "engine"
            binary.write_bytes(b"/System/Library/Frameworks/Metal.framework/Versions/A/Metal")
            self.assertIn("metal", setup_flow.binary_backends(binary))
            self.assertEqual(setup_flow.binary_backend(binary), "metal")
