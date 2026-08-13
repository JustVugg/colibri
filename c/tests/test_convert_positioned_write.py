import ast
import os
import tempfile
import threading
import types
import unittest
from pathlib import Path


CONVERTER = Path(__file__).resolve().parent.parent / "tools" / "convert_fp8_to_int4.py"


def load_positioned_write(os_module):
    tree = ast.parse(CONVERTER.read_text(encoding="utf-8"), filename=str(CONVERTER))
    helper = next(node for node in tree.body
                  if isinstance(node, ast.FunctionDef) and node.name == "_positioned_write")
    namespace = {"os": os_module, "_positioned_write_lock": threading.Lock()}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), str(CONVERTER), "exec"), namespace)
    return namespace["_positioned_write"]


class ShortWriteFallback:
    SEEK_SET = os.SEEK_SET
    pwrite = None

    @staticmethod
    def lseek(fd, offset, whence):
        return os.lseek(fd, offset, whence)

    @staticmethod
    def write(fd, data):
        return os.write(fd, data[:2])


class PositionedWriteTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "part"

    def open_part(self, size):
        self.path.write_bytes(b"\x00" * size)
        flags = os.O_WRONLY | getattr(os, "O_BINARY", 0)
        fd = os.open(self.path, flags)
        self.addCleanup(os.close, fd)
        return fd

    def test_fallback_writes_disjoint_binary_chunks_exactly(self):
        first = b"A\nB\x1aC"
        second = b"D\x1aE\nF"
        fd = self.open_part(len(first) + len(second))
        positioned_write = load_positioned_write(ShortWriteFallback)

        positioned_write(fd, second, len(first))
        positioned_write(fd, first, 0)

        self.assertEqual(self.path.read_bytes(), first + second)

    @unittest.skipUnless(hasattr(os, "pwrite"), "os.pwrite is unavailable")
    def test_pwrite_path_retries_short_writes(self):
        payload = b"A\nB\x1aC"
        fd = self.open_part(len(payload))

        def short_pwrite(descriptor, data, offset):
            return os.pwrite(descriptor, data[:2], offset)

        positioned_write = load_positioned_write(
            types.SimpleNamespace(pwrite=short_pwrite))
        positioned_write(fd, payload, 0)

        self.assertEqual(self.path.read_bytes(), payload)

    def test_zero_length_write_is_an_error(self):
        fd = self.open_part(1)
        cases = (
            (types.SimpleNamespace(pwrite=lambda descriptor, data, offset: 0),
             "pwrite returned zero bytes"),
            (types.SimpleNamespace(
                pwrite=None, SEEK_SET=os.SEEK_SET, lseek=os.lseek,
                write=lambda descriptor, data: 0),
             "write returned zero bytes"),
        )
        for os_module, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(OSError, message):
                load_positioned_write(os_module)(fd, b"x", 0)


if __name__ == "__main__":
    unittest.main()
