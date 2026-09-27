"""Inkling auto-sized its expert cache from mem_avail_bytes().

On Windows that function took the #else return 0 branch, so every
`coli serve --model <inkling>` with no --cap used 16 experts/layer
regardless of installed RAM, and /health reported ram_total_gb 0.0.
"""
import ctypes
import re
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENGINE_C = HERE.parent / "inkling.c"


def _c_function(source, signature):
    match = re.search(re.escape(signature) + r"\s*\{", source)
    if not match:
        raise ValueError("no definition for " + signature)
    start = match.start()
    brace = match.end() - 1
    depth = 0
    for index, char in enumerate(source[brace:], start=brace):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return source[start:index + 1]
    raise ValueError("unbalanced braces in " + signature)


def _windows_avail_phys():
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]
    stat = MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(stat)
    ok = ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
    return stat.ullAvailPhys if ok else 0


class InklingMemProbeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = ENGINE_C.read_text(encoding="utf-8")
        cls.avail = _c_function(cls.source, "static double mem_avail_bytes(void)")
        cls.hwinfo = _c_function(cls.source, "static void serve_hwinfo(Model *m)")

    def test_mem_avail_bytes_uses_the_shared_probe(self):
        self.assertIn("compat_mem_available_gb", self.avail)
        self.assertNotRegex(
            self.avail, r"#else\s*\n\s*return 0;",
            "Windows still takes a return-0 branch instead of measuring RAM")

    def test_serve_hwinfo_fills_ram_when_proc_is_missing(self):
        self.assertIn("compat_meminfo_gb", self.hwinfo)

    def test_windows_host_has_ram_the_old_probe_would_ignore(self):
        if sys.platform != "win32":
            return
        free = _windows_avail_phys()
        self.assertGreater(free, 1_000_000_000,
                           "this host reports no reclaimable RAM")
        if re.search(r"#else\s*\n\s*return 0;", self.avail):
            self.fail(
                "mem_avail_bytes() returns 0 on Windows; this host has "
                f"{free / 1e9:.1f} GB free, so auto cap would stay at "
                "16 experts/layer")
