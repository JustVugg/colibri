import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


C_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(C_DIR))

import ramdisk  # noqa: E402
from ramdisk_support import discovery, linux_ops  # noqa: E402
from ramdisk_support.platform_ops import (  # noqa: E402
    UNSUPPORTED_PLATFORM_REASON,
    UnsupportedPlatformOps,
    get_platform_ops,
)


class RamdiskPlatformTest(unittest.TestCase):
    FRESH_PROCESS_CONTRACT = r"""
import argparse
import builtins
import contextlib
import io
import json
import os
import sys
import urllib.request
from pathlib import Path

support_dir = sys.argv[1]
target_platform = sys.argv[2]
temporary_root = Path(sys.argv[3]).resolve()

missing_by_platform = {
    "win32": (
        "getuid",
        "geteuid",
        "getgid",
        "getgroups",
        "getpgid",
        "killpg",
        "statvfs",
        "sched_getaffinity",
    ),
    "darwin": ("sched_getaffinity",),
}
removed = []
for name in missing_by_platform[target_platform]:
    if hasattr(os, name):
        delattr(os, name)
        removed.append(name)
unavailable = [
    name for name in missing_by_platform[target_platform] if not hasattr(os, name)
]
assert unavailable == list(missing_by_platform[target_platform]), unavailable

real_open = builtins.open

def guarded_open(path, *args, **kwargs):
    try:
        spelling = os.fsdecode(os.fspath(path)).replace("\\", "/")
    except TypeError:
        spelling = ""
    if spelling in ("/proc", "/sys") or spelling.startswith(("/proc/", "/sys/")):
        raise AssertionError("unsupported platform probed Linux path " + spelling)
    return real_open(path, *args, **kwargs)

builtins.open = guarded_open
sys.platform = target_platform
sys.modules["fcntl"] = None
sys.path.insert(0, support_dir)

import ramdisk
from ramdisk_support.discovery import discover_hardware
from ramdisk_support.platform_ops import (
    UNSUPPORTED_PLATFORM_REASON,
    UnsupportedPlatformOps,
)

process_errors = {}
for name, operation in (
    ("identity", lambda: ramdisk._proc_identity(1)),
    ("group_alive", lambda: ramdisk._group_alive(1)),
    (
        "busy_mounts",
        lambda: ramdisk._busy_mount_references("/mnt/colibri"),
    ),
):
    try:
        operation()
    except ramdisk.RamdiskError as exc:
        process_errors[name] = str(exc)
    else:
        raise AssertionError(
            "%s unexpectedly used an unsupported process capability" % name
        )
assert process_errors == {
    "busy_mounts": UNSUPPORTED_PLATFORM_REASON,
    "identity": UNSUPPORTED_PLATFORM_REASON,
    "group_alive": UNSUPPORTED_PLATFORM_REASON,
}, process_errors

parser = argparse.ArgumentParser(prog="coli ramdisk")
ramdisk.configure_parser(parser)
help_text = parser.format_help()
assert "ACTION" in help_text
status_args = parser.parse_args(["status", "--json"])

os.environ["XDG_STATE_HOME"] = str(temporary_root / "state")
os.environ["COLI_RAMDISK_MANIFEST"] = str(temporary_root / "manifest.json")
report = ramdisk.status()
assert report["present"] is False, report

stdout = io.StringIO()
stderr = io.StringIO()
with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
    exit_code = ramdisk.dispatch(status_args)
payload = json.loads(stdout.getvalue())
assert exit_code == 0, exit_code
assert stderr.getvalue() == "", stderr.getvalue()
assert payload["schema"] == ramdisk.STATUS_SCHEMA, payload
assert payload["present"] is False, payload

hardware = discover_hardware(ops=UnsupportedPlatformOps(target_platform))
assert (
    hardware["capabilities"]["reason"] == UNSUPPORTED_PLATFORM_REASON
), hardware["capabilities"]

print(
    json.dumps(
        {
            "platform": target_platform,
            "removed": sorted(removed),
            "unavailable": sorted(unavailable),
            "plan_schema": ramdisk.PLAN_SCHEMA,
            "platform_reason": hardware["capabilities"]["reason"],
            "process_errors": process_errors,
            "status_schema": payload["schema"],
        },
        sort_keys=True,
    )
)
"""

    def _run_fresh_process_contract(self, platform_name):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    self.FRESH_PROCESS_CONTRACT,
                    str(C_DIR),
                    platform_name,
                    str(root),
                ],
                cwd=C_DIR,
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_win32_fresh_process_facade_stays_portable_without_posix_os_apis(self):
        result = self._run_fresh_process_contract("win32")

        self.assertEqual(result["platform"], "win32")
        self.assertEqual(
            set(result["unavailable"]),
            {
                "getuid",
                "geteuid",
                "getgid",
                "getgroups",
                "getpgid",
                "killpg",
                "sched_getaffinity",
                "statvfs",
            },
        )
        self.assertEqual(result["plan_schema"], ramdisk.PLAN_SCHEMA)
        self.assertEqual(result["platform_reason"], UNSUPPORTED_PLATFORM_REASON)
        self.assertEqual(
            set(result["process_errors"].values()),
            {UNSUPPORTED_PLATFORM_REASON},
        )
        self.assertEqual(result["status_schema"], ramdisk.STATUS_SCHEMA)

    def test_darwin_fresh_process_facade_stays_portable_without_linux_os_apis(self):
        result = self._run_fresh_process_contract("darwin")

        self.assertEqual(result["platform"], "darwin")
        self.assertEqual(result["unavailable"], ["sched_getaffinity"])
        self.assertEqual(result["plan_schema"], ramdisk.PLAN_SCHEMA)
        self.assertEqual(result["platform_reason"], UNSUPPORTED_PLATFORM_REASON)
        self.assertEqual(
            set(result["process_errors"].values()),
            {UNSUPPORTED_PLATFORM_REASON},
        )
        self.assertEqual(result["status_schema"], ramdisk.STATUS_SCHEMA)

    def test_import_does_not_probe_linux_facilities(self):
        script = r"""
import builtins
import os
import shutil
import sys

real_open = builtins.open

def guarded_open(path, *args, **kwargs):
    try:
        spelling = os.fsdecode(os.fspath(path)).replace("\\", "/")
    except TypeError:
        spelling = ""
    if spelling in ("/proc", "/sys") or spelling.startswith(("/proc/", "/sys/")):
        raise AssertionError("RAM-disk import probed " + spelling)
    return real_open(path, *args, **kwargs)

def reject_which(name, *args, **kwargs):
    raise AssertionError("RAM-disk import searched PATH for " + name)

builtins.open = guarded_open
shutil.which = reject_which
if hasattr(os, "sched_getaffinity"):
    del os.sched_getaffinity
sys.path.insert(0, sys.argv[1])
import ramdisk
"""
        result = subprocess.run(
            [sys.executable, "-c", script, str(C_DIR)],
            cwd=C_DIR,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_selector_returns_explicit_unsupported_capabilities(self):
        ops = get_platform_ops("darwin")

        self.assertIsInstance(ops, UnsupportedPlatformOps)
        self.assertEqual(
            ops.capabilities(),
            {
                "platform": "darwin",
                "hardware_discovery": False,
                "cgroup_memory": False,
                "numa": False,
                "ramdisk_lifecycle": False,
                "reason": UNSUPPORTED_PLATFORM_REASON,
            },
        )

    def test_facade_discovery_does_not_probe_linux_facilities_when_unsupported(self):
        ops = UnsupportedPlatformOps("win32")
        with (
            mock.patch.object(discovery, "get_platform_ops", return_value=ops),
            mock.patch(
                "builtins.open",
                side_effect=AssertionError("unsupported discovery read a host file"),
            ) as open_file,
            mock.patch.object(
                linux_ops.shutil,
                "which",
                side_effect=AssertionError("unsupported discovery searched PATH"),
            ) as which,
        ):
            hardware = ramdisk.discover_hardware()

        self.assertFalse(hardware["linux"])
        self.assertFalse(hardware["capabilities"]["hardware_discovery"])
        self.assertEqual(
            hardware["capabilities"]["reason"],
            UNSUPPORTED_PLATFORM_REASON,
        )
        self.assertFalse(hardware["tmpfs"]["supported"])
        self.assertIsNone(hardware["mount"])
        open_file.assert_not_called()
        which.assert_not_called()

    def test_synthetic_cgroup_contracts_remain_platform_independent(self):
        with tempfile.TemporaryDirectory() as temporary:
            mountpoint = Path(temporary).resolve() / "cgroup"
            leaf = mountpoint / "scope"
            leaf.mkdir(parents=True)
            (mountpoint / "memory.max").write_text("max", encoding="utf-8")
            (mountpoint / "memory.current").write_text("0", encoding="utf-8")
            (mountpoint / "memory.high").write_text("max", encoding="utf-8")
            (leaf / "memory.max").write_text("4096", encoding="utf-8")
            (leaf / "memory.current").write_text("1024", encoding="utf-8")
            (leaf / "memory.high").write_text("2048", encoding="utf-8")
            mountinfo_path = (
                mountpoint.as_posix()
                .replace("\\", "\\134")
                .replace(" ", "\\040")
                .replace("\t", "\\011")
                .replace("\n", "\\012")
            )

            with mock.patch.object(
                discovery,
                "get_platform_ops",
                return_value=UnsupportedPlatformOps("darwin"),
            ):
                result = ramdisk._discover_cgroup_memory(
                    cgroup_text="0::/scope\n",
                    mountinfo_text=(
                        "36 25 0:32 / %s rw,nosuid,nodev,noexec "
                        "- cgroup2 cgroup rw\n" % mountinfo_path
                    ),
                )

        self.assertEqual(result["status"], "limited")
        self.assertEqual(result["available_bytes"], 3072)
        self.assertEqual(result["high_available_bytes"], 1024)


if __name__ == "__main__":
    unittest.main()
