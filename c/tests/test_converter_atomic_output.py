import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


CONVERTER = Path(__file__).resolve().parent.parent / "tools" / "convert_fp8_to_int4.py"


def load_atomic_save():
    spec = importlib.util.spec_from_file_location("atomic_converter_test", CONVERTER)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, {"numpy": types.ModuleType("numpy")}):
        spec.loader.exec_module(module)
    return module._save_file_atomic


_save_file_atomic = load_atomic_save()


class AtomicConverterOutputTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.destination = Path(self.directory.name) / "out-00000.safetensors"
        self.temporary = Path(str(self.destination) + ".tmp")

    def test_success_publishes_complete_output(self):
        def save_file(tensors, path):
            Path(path).write_bytes(tensors["payload"])

        _save_file_atomic(save_file, {"payload": b"complete"}, self.destination)

        self.assertEqual(self.destination.read_bytes(), b"complete")
        self.assertFalse(self.temporary.exists())

    def test_failed_save_leaves_no_resumable_output(self):
        def save_file(tensors, path):
            Path(path).write_bytes(b"partial")
            raise OSError("disk full")

        with self.assertRaisesRegex(OSError, "disk full"):
            _save_file_atomic(save_file, {}, self.destination)

        self.assertFalse(self.destination.exists())
        self.assertFalse(self.temporary.exists())

    def test_failed_overwrite_preserves_previous_output(self):
        self.destination.write_bytes(b"previous")

        def save_file(tensors, path):
            Path(path).write_bytes(b"partial")
            raise KeyboardInterrupt

        with self.assertRaises(KeyboardInterrupt):
            _save_file_atomic(save_file, {}, self.destination)

        self.assertEqual(self.destination.read_bytes(), b"previous")
        self.assertFalse(self.temporary.exists())


if __name__ == "__main__":
    unittest.main()
