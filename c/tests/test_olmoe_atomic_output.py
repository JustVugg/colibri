import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


CONVERTER = Path(__file__).resolve().parent.parent / "tools" / "convert_olmoe_merged.py"


def load_converter():
    torch = types.ModuleType("torch")
    safetensors = types.ModuleType("safetensors")
    safetensors.safe_open = mock.Mock()
    safetensors_torch = types.ModuleType("safetensors.torch")
    safetensors_torch.save_file = mock.Mock()
    spec = importlib.util.spec_from_file_location("olmoe_atomic_output_test", CONVERTER)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, {
        # numpy joins the converter's dependency guard in #1048 (safetensors.torch
        # imports it internally). This test imports the converter with its real
        # dependencies absent, so every name in that guard has to be stubbed here
        # or the guard sys.exit()s during exec_module and the module never loads.
        "numpy": mock.Mock(),
        "torch": torch,
        "safetensors": safetensors,
        "safetensors.torch": safetensors_torch,
    }):
        spec.loader.exec_module(module)
    return module


converter = load_converter()


class FakeTensor:
    def __init__(self, payload=b"tensor"):
        self.payload = payload

    def contiguous(self):
        return self


class AtomicOlmoeOutputTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.out_dir = Path(self.directory.name)
        self.destination = self.out_dir / "model-00000.safetensors"
        self.temporary = Path(str(self.destination) + ".tmp")

    def test_success_publishes_complete_shard(self):
        writer = converter.OutputWriter(self.out_dir)
        writer.add("weight", FakeTensor(b"complete"))

        def save_file(tensors, path):
            Path(path).write_bytes(tensors["weight"].payload)

        with mock.patch.object(converter, "save_file", side_effect=save_file):
            writer.flush()

        self.assertEqual(self.destination.read_bytes(), b"complete")
        self.assertFalse(self.temporary.exists())
        self.assertEqual(writer.buf, {})
        self.assertEqual(writer.shard_idx, 1)

    def test_failed_save_cleans_temp_and_preserves_retry_state(self):
        writer = converter.OutputWriter(self.out_dir)
        tensor = FakeTensor()
        writer.add("weight", tensor)
        self.temporary.write_bytes(b"stale")

        def fail_save(tensors, path):
            self.assertFalse(Path(path).exists())
            Path(path).write_bytes(b"partial")
            raise OSError("disk full")

        with mock.patch.object(converter, "save_file", side_effect=fail_save):
            with self.assertRaisesRegex(OSError, "disk full"):
                writer.flush()

        self.assertFalse(self.destination.exists())
        self.assertFalse(self.temporary.exists())
        self.assertIs(writer.buf["weight"], tensor)
        self.assertEqual(writer.shard_idx, 0)

    def test_grouped_add_keeps_expert_companions_in_same_shard(self):
        writer = converter.OutputWriter(self.out_dir, flush_every=3)
        published = []

        def save_file(tensors, path):
            published.append(set(tensors))
            Path(path).write_bytes(b"complete")

        with mock.patch.object(converter, "save_file", side_effect=save_file):
            writer.add("dense.a", FakeTensor())
            writer.add("dense.b", FakeTensor())
            writer.add_many((
                ("expert.merged_weight", FakeTensor()),
                ("expert.qs", FakeTensor()),
            ))

        self.assertEqual(len(published), 1)
        self.assertIn("expert.merged_weight", published[0])
        self.assertIn("expert.qs", published[0])
        self.assertEqual(writer.buf, {})


if __name__ == "__main__":
    unittest.main()
