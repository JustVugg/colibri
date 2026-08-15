import ast
import builtins
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


CONVERTER = Path(__file__).resolve().parent.parent / "tools" / "convert_inkling_int4.py"
ERROR = "ERROR: another converter is already using this output directory."


def load_lock_helper():
    tree = ast.parse(CONVERTER.read_text(encoding="utf-8"), filename=str(CONVERTER))
    helper = next(node for node in tree.body
                  if isinstance(node, ast.FunctionDef) and node.name == "acquire_output_lock")
    namespace = {"os": os, "sys": sys}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), str(CONVERTER), "exec"), namespace)
    return namespace["acquire_output_lock"]


class InklingConverterLockTest(unittest.TestCase):
    def setUp(self):
        self.acquire_output_lock = load_lock_helper()
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def test_unix_uses_nonblocking_flock(self):
        fcntl = types.SimpleNamespace(LOCK_EX=2, LOCK_NB=4, flock=mock.Mock())
        with mock.patch.dict(sys.modules, {"fcntl": fcntl}):
            lock = self.acquire_output_lock(self.directory.name)
        self.addCleanup(lock.close)
        fcntl.flock.assert_called_once_with(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_windows_uses_nonblocking_msvcrt_lock(self):
        msvcrt = types.SimpleNamespace(LK_NBLCK=3, locking=mock.Mock())
        real_import = builtins.__import__

        def platform_import(name, *args, **kwargs):
            if name == "fcntl":
                raise ImportError("fcntl is unavailable")
            if name == "msvcrt":
                return msvcrt
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=platform_import):
            lock = self.acquire_output_lock(self.directory.name)
        self.addCleanup(lock.close)
        msvcrt.locking.assert_called_once_with(lock.fileno(), msvcrt.LK_NBLCK, 1)

    def test_lock_contention_keeps_existing_error(self):
        fcntl = types.SimpleNamespace(
            LOCK_EX=2, LOCK_NB=4, flock=mock.Mock(side_effect=BlockingIOError))
        with mock.patch.dict(sys.modules, {"fcntl": fcntl}), \
             self.assertRaisesRegex(SystemExit, ERROR):
            self.acquire_output_lock(self.directory.name)


if __name__ == "__main__":
    unittest.main()
