"""machine_info() must report the machine's real RAM on win32 (#1042).

ram_gb sizes the eviction write in evict_cache(): a hardcoded 8.0 on a
128 GB box writes 9 GB, evicts nothing, and the run labelled "cold" is
measured warm and published as cold. These tests run on any host by
mocking sys.platform and the two win32 probes.
"""
import ctypes
import sys
import unittest
from unittest import mock

from tools import datapoint

GB = 1073741824


class MachineInfoWin32Test(unittest.TestCase):
    def _win32_info(self, memstatus_ok, total_phys=128 * GB):
        def fake_memstatus(argp):
            if not memstatus_ok:
                return 0
            argp._obj.ullTotalPhys = total_phys
            return 1

        windll = mock.MagicMock()
        windll.kernel32.GlobalMemoryStatusEx.side_effect = fake_memstatus
        with mock.patch.object(sys, "platform", "win32"), \
             mock.patch.object(ctypes, "windll", windll, create=True):
            return datapoint.machine_info()

    def test_win32_reports_real_ram(self):
        info = self._win32_info(memstatus_ok=True)
        self.assertAlmostEqual(info["ram_gb"], 128.0)
        self.assertEqual(info["ram"], "128 GB")
        self.assertTrue(info["os"].startswith("Windows"))

    def test_win32_probe_failure_falls_back(self):
        """A failed probe keeps the old conservative default rather than 0.0 —
        an eviction write sized from 0 GB would silently evict nothing."""
        info = self._win32_info(memstatus_ok=False)
        self.assertEqual(info["ram_gb"], 8.0)
        self.assertEqual(info["ram"], "?")

    def test_unknown_platform_keeps_fallback(self):
        with mock.patch.object(sys, "platform", "freebsd14"):
            info = datapoint.machine_info()
        self.assertEqual(info["ram_gb"], 8.0)
        self.assertEqual(info["ram"], "?")


if __name__ == "__main__":
    unittest.main()
