import argparse
import errno
import io
import json
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


C_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(C_DIR))

import ramdisk  # noqa: E402
from ramdisk_support import discovery, linux_ops, processes  # noqa: E402
from ramdisk_support.platform_ops import (  # noqa: E402
    UNSUPPORTED_PLATFORM_REASON,
    UnsupportedPlatformOps,
    get_platform_ops,
)

if __package__:
    from .platform_test_support import (  # noqa: E402
        PLATFORM_SKIP_INVENTORY,
        assert_platform_skip_inventory,
    )
else:
    from platform_test_support import (  # noqa: E402
        PLATFORM_SKIP_INVENTORY,
        assert_platform_skip_inventory,
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
sys.modules["fcntl"] = None
sys.path.insert(0, support_dir)

import ramdisk
from ramdisk_support.platform_ops import (
    UNSUPPORTED_PLATFORM_REASON,
    UnsupportedPlatformOps,
)

eager_optional_modules = sorted(
    name
    for name in (
        "ramdisk_support.benchmark",
        "ramdisk_support.curses_ui",
        "ramdisk_support.runtime_monitor",
        "ramdisk_ui",
        "ramdisk_textual",
        "ssl",
        "urllib.request",
    )
    if name in sys.modules
)
assert eager_optional_modules == [], eager_optional_modules

ops = UnsupportedPlatformOps(target_platform)
ramdisk.get_platform_ops = lambda platform_name=None: ops
process_errors = {}
for name, operation in (
    ("identity", lambda: ramdisk._proc_identity(1, ops=ops)),
    ("group_alive", lambda: ramdisk._group_alive(1, ops=ops)),
    (
        "busy_mounts",
        lambda: ramdisk._busy_mount_references("/mnt/colibri", ops=ops),
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
stop_args = parser.parse_args(["stop"])

os.environ["XDG_STATE_HOME"] = str(temporary_root / "state")
os.environ["COLI_RAMDISK_MANIFEST"] = str(temporary_root / "manifest.json")
report = ramdisk.status()
assert report["present"] is False, report

def forbidden(label):
    def fail(*args, **kwargs):
        del args, kwargs
        raise AssertionError("unsupported mutator reached " + label)
    return fail

forbidden_names = (
    "_lifecycle_lock",
    "_load_manifest",
    "_save_manifest",
    "_trusted_system_binary",
    "_confirm",
    "_mount_tmpfs",
    "_umount_path",
    "_resolve_engine_path",
)
originals = {name: getattr(ramdisk, name) for name in forbidden_names}
for name in forbidden_names:
    setattr(ramdisk, name, forbidden(name))

mutating_errors = {}
empty_args = argparse.Namespace()
for name, operation in (
    ("prepare", lambda: ramdisk.prepare(empty_args)),
    ("start", lambda: ramdisk.start(empty_args)),
    ("stop", lambda: ramdisk.stop()),
    ("benchmark", lambda: ramdisk.benchmark(empty_args)),
    ("destroy", lambda: ramdisk.destroy(empty_args)),
):
    try:
        operation()
    except ramdisk.RamdiskError as exc:
        mutating_errors[name] = str(exc)
    else:
        raise AssertionError(name + " unexpectedly ran on " + target_platform)
assert mutating_errors == {
    "benchmark": UNSUPPORTED_PLATFORM_REASON,
    "destroy": UNSUPPORTED_PLATFORM_REASON,
    "prepare": UNSUPPORTED_PLATFORM_REASON,
    "start": UNSUPPORTED_PLATFORM_REASON,
    "stop": UNSUPPORTED_PLATFORM_REASON,
}, mutating_errors
for name, original in originals.items():
    setattr(ramdisk, name, original)
assert not (temporary_root / "state").exists()
assert not (temporary_root / "manifest.json").exists()
assert not any(
    name in sys.modules
    for name in (
        "ramdisk_support.benchmark",
        "ramdisk_support.curses_ui",
        "ramdisk_support.runtime_monitor",
        "ramdisk_ui",
        "ramdisk_textual",
    )
), sorted(sys.modules)

stdout = io.StringIO()
stderr = io.StringIO()
with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
    exit_code = ramdisk.dispatch(status_args)
payload = json.loads(stdout.getvalue())
assert exit_code == 0, exit_code
assert stderr.getvalue() == "", stderr.getvalue()
assert payload["schema"] == ramdisk.STATUS_SCHEMA, payload
assert payload["present"] is False, payload

stdout = io.StringIO()
stderr = io.StringIO()
with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
    stop_exit_code = ramdisk.dispatch(stop_args)
assert stop_exit_code == 2, stop_exit_code
assert stdout.getvalue() == "", stdout.getvalue()
assert stderr.getvalue() == (
    "coli ramdisk: " + UNSUPPORTED_PLATFORM_REASON + "\n"
), stderr.getvalue()

hardware = ramdisk._discover_hardware(ops=ops)
assert (
    hardware["capabilities"]["reason"] == UNSUPPORTED_PLATFORM_REASON
), hardware["capabilities"]

print(
    json.dumps(
        {
            "platform": target_platform,
            "eager_optional_modules": eager_optional_modules,
            "removed": sorted(removed),
            "unavailable": sorted(unavailable),
            "plan_schema": ramdisk.PLAN_SCHEMA,
            "platform_reason": hardware["capabilities"]["reason"],
            "process_errors": process_errors,
            "mutating_errors": mutating_errors,
            "stop_error": UNSUPPORTED_PLATFORM_REASON,
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
        self.assertEqual(result["eager_optional_modules"], [])
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
        self.assertEqual(
            set(result["mutating_errors"]),
            {"prepare", "start", "stop", "benchmark", "destroy"},
        )
        self.assertEqual(
            set(result["mutating_errors"].values()),
            {UNSUPPORTED_PLATFORM_REASON},
        )
        self.assertEqual(result["stop_error"], UNSUPPORTED_PLATFORM_REASON)
        self.assertEqual(result["status_schema"], ramdisk.STATUS_SCHEMA)

    def test_darwin_fresh_process_facade_stays_portable_without_linux_os_apis(self):
        result = self._run_fresh_process_contract("darwin")

        self.assertEqual(result["platform"], "darwin")
        self.assertEqual(result["eager_optional_modules"], [])
        self.assertEqual(result["unavailable"], ["sched_getaffinity"])
        self.assertEqual(result["plan_schema"], ramdisk.PLAN_SCHEMA)
        self.assertEqual(result["platform_reason"], UNSUPPORTED_PLATFORM_REASON)
        self.assertEqual(
            set(result["process_errors"].values()),
            {UNSUPPORTED_PLATFORM_REASON},
        )
        self.assertEqual(
            set(result["mutating_errors"]),
            {"prepare", "start", "stop", "benchmark", "destroy"},
        )
        self.assertEqual(
            set(result["mutating_errors"].values()),
            {UNSUPPORTED_PLATFORM_REASON},
        )
        self.assertEqual(result["stop_error"], UNSUPPORTED_PLATFORM_REASON)
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

    def test_process_mutators_reject_missing_control_before_lock_or_state(self):
        ops = mock.Mock(
            is_linux=True,
            process_control_supported=False,
        )
        forbidden_lock = mock.Mock(
            side_effect=AssertionError("lifecycle lock was reached")
        )
        empty_args = argparse.Namespace()

        with mock.patch.object(
            ramdisk, "get_platform_ops", return_value=ops
        ), mock.patch.object(
            ramdisk, "_lifecycle_lock", forbidden_lock
        ):
            for name, operation in (
                ("start", lambda: ramdisk.start(empty_args)),
                ("stop", lambda: ramdisk.stop()),
                ("benchmark", lambda: ramdisk.benchmark(empty_args)),
                ("destroy", lambda: ramdisk.destroy(empty_args)),
            ):
                with self.subTest(operation=name), self.assertRaisesRegex(
                    ramdisk.RamdiskError,
                    UNSUPPORTED_PLATFORM_REASON,
                ):
                    operation()

        forbidden_lock.assert_not_called()

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

    def test_platform_skip_inventory_is_exact_and_drift_checked(self):
        self.assertEqual(
            len(PLATFORM_SKIP_INVENTORY["linux_operational"]["tests"]),
            36,
        )
        self.assertEqual(
            len(PLATFORM_SKIP_INVENTORY["sigterm_handler"]["tests"]),
            6,
        )
        self.assertEqual(
            len(PLATFORM_SKIP_INVENTORY["sigint_handler"]["tests"]),
            1,
        )
        self.assertEqual(
            len(PLATFORM_SKIP_INVENTORY["linux_pidfd"]["tests"]),
            1,
        )
        assert_platform_skip_inventory()


class LinuxOperationalReadContractTest(unittest.TestCase):
    def test_trusted_helper_rejects_foreign_group_writable_parent(self):
        safe_directory = mock.Mock(
            st_mode=stat.S_IFDIR | 0o755,
            st_uid=0,
            st_gid=0,
        )
        unsafe_directory = mock.Mock(
            st_mode=stat.S_IFDIR | 0o775,
            st_uid=0,
            st_gid=4321,
        )
        safe_executable = mock.Mock(
            st_mode=stat.S_IFREG | 0o755,
            st_uid=0,
            st_gid=0,
        )

        def file_stat(path):
            if path == "/":
                return safe_directory
            if path == "/foreign/bin/fuser":
                return safe_executable
            if path == "/foreign/bin":
                return unsafe_directory
            if path == "/foreign":
                return safe_directory
            raise FileNotFoundError(errno.ENOENT, "not found", path)

        with mock.patch.object(
            linux_ops, "_require_linux"
        ), mock.patch.object(
            linux_ops.shutil,
            "which",
            return_value="/foreign/bin/fuser",
        ), mock.patch.object(
            linux_ops.os.path,
            "realpath",
            side_effect=lambda path: path,
        ), mock.patch.object(
            linux_ops.os, "stat", side_effect=file_stat
        ), mock.patch.object(
            linux_ops.os, "access", return_value=False
        ), mock.patch.object(
            linux_ops, "current_euid", return_value=1000
        ):
            with self.assertRaisesRegex(
                ramdisk.RamdiskError,
                "rejected writable candidates: /foreign/bin/fuser",
            ):
                linux_ops._trusted_system_binary("fuser")

    def test_trusted_helper_accepts_sticky_nix_store_ancestor(self):
        safe_directory = mock.Mock(
            st_mode=stat.S_IFDIR | 0o555,
            st_uid=0,
            st_gid=0,
        )
        root_directory = mock.Mock(
            st_mode=stat.S_IFDIR | 0o755,
            st_uid=0,
            st_gid=0,
        )
        nix_store = mock.Mock(
            st_mode=stat.S_IFDIR | stat.S_ISVTX | 0o775,
            st_uid=0,
            st_gid=30000,
        )
        safe_executable = mock.Mock(
            st_mode=stat.S_IFREG | 0o555,
            st_uid=0,
            st_gid=0,
        )
        executable = "/nix/store/abc123-psmisc/bin/fuser"

        def file_stat(path):
            if path in ("/", "/nix"):
                return root_directory
            if path == "/nix/store":
                return nix_store
            if path in (
                "/nix/store/abc123-psmisc",
                "/nix/store/abc123-psmisc/bin",
            ):
                return safe_directory
            if path == executable:
                return safe_executable
            raise FileNotFoundError(errno.ENOENT, "not found", path)

        with mock.patch.object(
            linux_ops, "_require_linux"
        ), mock.patch.object(
            linux_ops.shutil, "which", return_value=executable
        ), mock.patch.object(
            linux_ops.os.path,
            "realpath",
            side_effect=lambda path: path,
        ), mock.patch.object(
            linux_ops.os, "stat", side_effect=file_stat
        ), mock.patch.object(
            linux_ops.os, "access", return_value=False
        ), mock.patch.object(
            linux_ops, "current_euid", return_value=1000
        ):
            self.assertEqual(
                linux_ops._trusted_system_binary("fuser"),
                executable,
            )

    def test_process_group_scan_rejects_proc_enumeration_failures(self):
        failures = (
            PermissionError(errno.EACCES, "permission denied", "/proc"),
            OSError(errno.EIO, "input/output error", "/proc"),
        )
        for failure in failures:
            with self.subTest(errno=failure.errno), mock.patch.object(
                linux_ops, "_require_linux"
            ), mock.patch.object(
                linux_ops.os, "listdir", side_effect=failure
            ):
                with self.assertRaisesRegex(
                    ramdisk.RamdiskError,
                    "cannot enumerate Linux process table",
                ) as raised:
                    linux_ops._process_group_member_pids(731)
                self.assertIn(
                    "managed cleanup requires complete process-table visibility",
                    str(raised.exception),
                )

    def test_process_group_scan_rejects_unreadable_member_stat(self):
        failures = (
            PermissionError(
                errno.EACCES,
                "permission denied",
                "/proc/731/stat",
            ),
            OSError(errno.EIO, "input/output error", "/proc/731/stat"),
        )
        for failure in failures:
            with self.subTest(errno=failure.errno), mock.patch.object(
                linux_ops, "_require_linux"
            ), mock.patch.object(
                linux_ops.os, "listdir", return_value=["731"]
            ), mock.patch(
                "builtins.open", side_effect=failure
            ):
                with self.assertRaisesRegex(
                    ramdisk.RamdiskError,
                    "cannot read Linux process identity /proc/731/stat",
                ):
                    linux_ops._process_group_member_pids(731)

    def test_process_group_scan_rejects_truncated_member_stat(self):
        with mock.patch.object(
            linux_ops, "_require_linux"
        ), mock.patch.object(
            linux_ops.os, "listdir", return_value=["731"]
        ), mock.patch(
            "builtins.open",
            mock.mock_open(read_data="731 (worker) S 1"),
        ):
            with self.assertRaisesRegex(
                ramdisk.RamdiskError,
                "cannot parse Linux process identity /proc/731/stat",
            ):
                linux_ops._process_group_member_pids(731)

    def test_process_group_scan_accepts_non_utf8_process_name_bytes(self):
        raw_stat = b"731 (worker-\xff) S 1 731 731 0 -1 0\n"

        def open_proc(path, mode, *, encoding, errors, newline):
            self.assertEqual(path, "/proc/731/stat")
            self.assertEqual(mode, "r")
            self.assertEqual(newline, "")
            return io.TextIOWrapper(
                io.BytesIO(raw_stat),
                encoding=encoding,
                errors=errors,
                newline=newline,
            )

        with mock.patch.object(
            linux_ops, "_require_linux"
        ), mock.patch.object(
            linux_ops.os, "listdir", return_value=["731"]
        ), mock.patch(
            "builtins.open", side_effect=open_proc
        ):
            self.assertEqual(
                linux_ops._process_group_member_pids(731),
                [731],
            )

    def test_process_group_scan_preserves_pid_disappearance(self):
        with mock.patch.object(
            linux_ops, "_require_linux"
        ), mock.patch.object(
            linux_ops.os, "listdir", return_value=["731"]
        ), mock.patch(
            "builtins.open",
            side_effect=FileNotFoundError(
                errno.ENOENT,
                "no such process",
                "/proc/731/stat",
            ),
        ):
            self.assertEqual(
                linux_ops._process_group_member_pids(731),
                [],
            )

    def test_empty_dead_process_group_is_not_running(self):
        ops = mock.Mock()
        ops.process_group_member_pids.return_value = []
        ops.process_group_alive.return_value = False

        members = processes._process_group_members(
            731,
            ops=ops,
            proc_identity=mock.Mock(),
        )
        result = processes._process_matches(
            {"pid": 731, "pgid": 731},
            proc_identity=lambda ignored: None,
            process_group_members=lambda ignored: members,
        )

        self.assertEqual(members, ([], []))
        self.assertEqual(result, (False, "not-running", None))

    def test_empty_live_process_group_is_unverified(self):
        ops = mock.Mock()
        ops.process_group_member_pids.return_value = []
        ops.process_group_alive.return_value = True

        members = processes._process_group_members(
            731,
            ops=ops,
            proc_identity=mock.Mock(),
        )
        result = processes._process_matches(
            {"pid": 731, "pgid": 731},
            proc_identity=lambda ignored: None,
            process_group_members=lambda ignored: members,
        )

        self.assertEqual(members, ([], [731]))
        self.assertFalse(result[0])
        self.assertEqual(result[1], "unverified-process-group")

    def test_mount_table_rejects_operational_read_failures(self):
        failures = (
            PermissionError(
                errno.EACCES,
                "permission denied",
                "/proc/self/mountinfo",
            ),
            OSError(
                errno.EIO,
                "input/output error",
                "/proc/self/mountinfo",
            ),
        )
        for failure in failures:
            with self.subTest(errno=failure.errno), mock.patch.object(
                linux_ops, "_require_linux"
            ), mock.patch(
                "builtins.open", side_effect=failure
            ):
                with self.assertRaisesRegex(
                    ramdisk.RamdiskError,
                    "cannot read Linux mount table",
                ):
                    linux_ops._mount_table()

    def test_mount_table_rejects_malformed_records(self):
        with mock.patch.object(
            linux_ops, "_require_linux"
        ), mock.patch(
            "builtins.open",
            mock.mock_open(read_data="36 25 truncated\n"),
        ):
            with self.assertRaisesRegex(
                ramdisk.RamdiskError,
                "cannot parse Linux mount table .* line 1",
            ):
                linux_ops._mount_table()

    def test_mount_table_preserves_successfully_read_empty_table(self):
        with mock.patch.object(
            linux_ops, "_require_linux"
        ), mock.patch(
            "builtins.open", return_value=io.StringIO("")
        ):
            self.assertEqual(linux_ops._mount_table(), [])

    def test_busy_mount_scan_rejects_proc_enumeration_failure(self):
        denied = PermissionError(errno.EACCES, "permission denied", "/proc")
        with mock.patch.object(
            linux_ops, "_require_linux"
        ), mock.patch.object(
            linux_ops.os, "listdir", side_effect=denied
        ):
            with self.assertRaisesRegex(
                ramdisk.RamdiskError,
                "managed cleanup requires complete /proc visibility",
            ):
                linux_ops._busy_mount_references_proc("/mnt/colibri")

    def test_busy_mount_scan_rejects_unreadable_pid_reference(self):
        denied = PermissionError(
            errno.EACCES,
            "permission denied",
            "/proc/731/cwd",
        )
        with mock.patch.object(
            linux_ops, "_require_linux"
        ), mock.patch.object(
            linux_ops.os, "listdir", return_value=["731"]
        ), mock.patch.object(
            linux_ops.os, "readlink", side_effect=denied
        ):
            with self.assertRaisesRegex(
                ramdisk.RamdiskError,
                "cannot read process reference /proc/731/cwd.*hidepid",
            ):
                linux_ops._busy_mount_references_proc("/mnt/colibri")

    def test_busy_mount_scan_rejects_unreadable_or_malformed_maps(self):
        failures = (
            OSError(errno.EIO, "input/output error", "/proc/731/maps"),
            None,
        )
        for failure in failures:
            opened = (
                {"side_effect": failure}
                if failure is not None
                else {"return_value": io.StringIO("truncated\n")}
            )
            with self.subTest(failure=type(failure).__name__), mock.patch.object(
                linux_ops, "_require_linux"
            ), mock.patch.object(
                linux_ops.os, "listdir", return_value=["731"]
            ), mock.patch.object(
                linux_ops.os, "readlink", return_value="/"
            ), mock.patch(
                "builtins.open", **opened
            ):
                with self.assertRaisesRegex(
                    ramdisk.RamdiskError,
                    "process mappings /proc/731/maps",
                ):
                    linux_ops._busy_mount_references_proc("/mnt/colibri")

    def test_busy_mount_scan_rejects_unreadable_descriptor_table(self):
        def list_directory(path):
            if path == "/proc":
                return ["731"]
            raise PermissionError(errno.EACCES, "permission denied", path)

        with mock.patch.object(
            linux_ops, "_require_linux"
        ), mock.patch.object(
            linux_ops.os, "listdir", side_effect=list_directory
        ), mock.patch.object(
            linux_ops.os, "readlink", return_value="/"
        ), mock.patch(
            "builtins.open", return_value=io.StringIO("")
        ):
            with self.assertRaisesRegex(
                ramdisk.RamdiskError,
                "cannot enumerate process descriptors /proc/731/fd",
            ):
                linux_ops._busy_mount_references_proc("/mnt/colibri")

    def test_busy_mount_scan_preserves_per_pid_disappearance(self):
        vanished = FileNotFoundError(
            errno.ENOENT,
            "process exited",
            "/proc/731/cwd",
        )
        with mock.patch.object(
            linux_ops, "_require_linux"
        ), mock.patch.object(
            linux_ops.os, "listdir", return_value=["731"]
        ), mock.patch.object(
            linux_ops.os, "readlink", side_effect=vanished
        ), mock.patch(
            "builtins.open",
            side_effect=FileNotFoundError(
                errno.ENOENT,
                "process exited",
                "/proc/731/stat",
            ),
        ):
            self.assertEqual(
                linux_ops._busy_mount_references_proc("/mnt/colibri"),
                [],
            )

    def test_busy_mount_scan_rejects_live_pid_with_missing_endpoint(self):
        live_stat = "731 (worker) S 1 731 731 0 -1 0\n"

        for missing_leaf in ("cwd", "maps", "fd"):
            def list_directory(path):
                if path == "/proc":
                    return ["731"]
                if path == "/proc/731/fd":
                    if missing_leaf == "fd":
                        raise FileNotFoundError(
                            errno.ENOENT,
                            "leader endpoint unavailable",
                            path,
                        )
                    return []
                raise AssertionError("unexpected listdir %s" % path)

            def read_link(path):
                if path == "/proc/731/cwd" and missing_leaf == "cwd":
                    raise FileNotFoundError(
                        errno.ENOENT,
                        "leader endpoint unavailable",
                        path,
                    )
                return "/"

            def open_proc(path, *args, **kwargs):
                del args, kwargs
                if path == "/proc/731/stat":
                    return io.StringIO(live_stat)
                if path == "/proc/731/maps":
                    if missing_leaf == "maps":
                        raise FileNotFoundError(
                            errno.ENOENT,
                            "leader endpoint unavailable",
                            path,
                        )
                    return io.StringIO("")
                raise AssertionError("unexpected open %s" % path)

            with self.subTest(endpoint=missing_leaf), mock.patch.object(
                linux_ops, "_require_linux"
            ), mock.patch.object(
                linux_ops.os, "listdir", side_effect=list_directory
            ), mock.patch.object(
                linux_ops.os, "readlink", side_effect=read_link
            ), mock.patch(
                "builtins.open", side_effect=open_proc
            ):
                with self.assertRaisesRegex(
                    ramdisk.RamdiskError,
                    "missing process endpoint /proc/731/%s.*PID 731 remains"
                    % missing_leaf,
                ):
                    linux_ops._busy_mount_references_proc("/mnt/colibri")

    def test_busy_mount_scan_skips_kernel_thread_with_missing_cwd(self):
        kernel_thread_stat = (
            "731 (kworker/0:1) I 2 0 0 0 -1 2097152\n"
        )
        vanished_cwd = FileNotFoundError(
            errno.ENOENT,
            "kernel threads have no cwd",
            "/proc/731/cwd",
        )

        with mock.patch.object(
            linux_ops, "_require_linux"
        ), mock.patch.object(
            linux_ops.os, "listdir", return_value=["731"]
        ), mock.patch.object(
            linux_ops.os, "readlink", side_effect=vanished_cwd
        ), mock.patch(
            "builtins.open", return_value=io.StringIO(kernel_thread_stat)
        ):
            self.assertEqual(
                linux_ops._busy_mount_references_proc("/mnt/colibri"),
                [],
            )

    def test_busy_mount_scan_rejects_zombie_leader_with_live_sibling(self):
        zombie_leader_stat = (
            "731 (worker) Z 1 731 731 0 -1 0 "
            "0 0 0 0 0 0 0 0 20 0 2\n"
        )
        vanished_cwd = FileNotFoundError(
            errno.ENOENT,
            "group leader exited",
            "/proc/731/cwd",
        )

        def list_directory(path):
            if path == "/proc":
                return ["731"]
            if path == "/proc/731/task":
                return ["731", "732"]
            raise AssertionError("unexpected listdir %s" % path)

        with mock.patch.object(
            linux_ops, "_require_linux"
        ), mock.patch.object(
            linux_ops.os, "listdir", side_effect=list_directory
        ), mock.patch.object(
            linux_ops.os, "readlink", side_effect=vanished_cwd
        ), mock.patch(
            "builtins.open", return_value=io.StringIO(zombie_leader_stat)
        ):
            with self.assertRaisesRegex(
                ramdisk.RamdiskError,
                "zombie/dead leader PID 731 still has sibling tasks 732",
            ):
                linux_ops._busy_mount_references_proc("/mnt/colibri")

    def test_busy_mount_scan_skips_proven_lone_zombie_leader(self):
        zombie_leader_stat = (
            "731 (worker) Z 1 731 731 0 -1 0 "
            "0 0 0 0 0 0 0 0 20 0 1\n"
        )

        def list_directory(path):
            return ["731"]

        with mock.patch.object(
            linux_ops, "_require_linux"
        ), mock.patch.object(
            linux_ops.os, "listdir", side_effect=list_directory
        ), mock.patch.object(
            linux_ops.os,
            "readlink",
            side_effect=FileNotFoundError(errno.ENOENT, "exited"),
        ), mock.patch(
            "builtins.open", return_value=io.StringIO(zombie_leader_stat)
        ):
            self.assertEqual(
                linux_ops._busy_mount_references_proc("/mnt/colibri"),
                [],
            )

    def test_busy_mount_scan_rejects_incomplete_zombie_task_snapshot(self):
        cases = (
            (1, []),
            (2, ["731"]),
            (1, ["732"]),
        )
        for num_threads, task_entries in cases:
            zombie_leader_stat = (
                "731 (worker) Z 1 731 731 0 -1 0 "
                "0 0 0 0 0 0 0 0 20 0 %d\n" % num_threads
            )

            def list_directory(path):
                if path == "/proc":
                    return ["731"]
                if path == "/proc/731/task":
                    return task_entries
                raise AssertionError("unexpected listdir %s" % path)

            with self.subTest(
                num_threads=num_threads,
                task_entries=task_entries,
            ), mock.patch.object(
                linux_ops, "_require_linux"
            ), mock.patch.object(
                linux_ops.os, "listdir", side_effect=list_directory
            ), mock.patch.object(
                linux_ops.os,
                "readlink",
                side_effect=FileNotFoundError(errno.ENOENT, "exited"),
            ), mock.patch(
                "builtins.open",
                return_value=io.StringIO(zombie_leader_stat),
            ):
                with self.assertRaisesRegex(
                    ramdisk.RamdiskError,
                    "incomplete process task snapshot",
                ):
                    linux_ops._busy_mount_references_proc("/mnt/colibri")

    def test_busy_mount_scan_decodes_procfs_mapping_path_escapes(self):
        mount_root = "/mnt/coli\\bri\tline\nroot"
        mapped = (mount_root + "/model.bin").replace(
            "\\", "\\134"
        ).replace("\t", "\\011").replace("\n", "\\012")
        maps = "1000-2000 r--p 00000000 00:01 7 %s\n" % mapped
        with mock.patch.object(
            linux_ops, "_require_linux"
        ), mock.patch.object(
            linux_ops.os, "listdir", return_value=["731"]
        ), mock.patch.object(
            linux_ops.os, "readlink", return_value="/"
        ), mock.patch(
            "builtins.open", return_value=io.StringIO(maps)
        ):
            self.assertEqual(
                linux_ops._busy_mount_references_proc(mount_root),
                [731],
            )

    def test_busy_mount_scan_finds_private_nonleader_task_references(self):
        live_stat = (
            "731 (worker) S 1 731 731 0 -1 0 "
            "0 0 0 0 0 0 0 0 20 0 2 0 100\n"
        )
        mount_root = "/mnt/colibri"
        mapped = "1000-2000 r--p 00000000 00:01 7 %s/model.bin\n" % mount_root

        for reference in ("cwd", "root", "maps", "fd"):
            def list_directory(path):
                if path == "/proc":
                    return ["731"]
                if path == "/proc/731/fd":
                    return []
                if path == "/proc/731/task":
                    return ["731", "732"]
                if path == "/proc/731/task/732/fd":
                    return ["9"] if reference == "fd" else []
                raise AssertionError("unexpected listdir %s" % path)

            def read_link(path):
                if path == "/proc/731/task/732/%s" % reference and reference in (
                    "cwd",
                    "root",
                ):
                    return mount_root + "/weights"
                if path == "/proc/731/task/732/fd/9":
                    return mount_root + "/weights"
                return "/"

            def open_proc(path, *args, **kwargs):
                del args, kwargs
                if path == "/proc/731/stat":
                    return io.StringIO(live_stat)
                if path == "/proc/731/maps":
                    return io.StringIO("")
                if path == "/proc/731/task/732/maps":
                    return io.StringIO(mapped if reference == "maps" else "")
                raise AssertionError("unexpected open %s" % path)

            with self.subTest(reference=reference), mock.patch.object(
                linux_ops, "_require_linux"
            ), mock.patch.object(
                linux_ops.os, "listdir", side_effect=list_directory
            ), mock.patch.object(
                linux_ops.os, "readlink", side_effect=read_link
            ), mock.patch(
                "builtins.open", side_effect=open_proc
            ):
                self.assertEqual(
                    linux_ops._busy_mount_references_proc(mount_root),
                    [731],
                )

    def test_busy_mount_scan_rejects_incomplete_live_task_snapshot(self):
        live_stat = (
            "731 (worker) S 1 731 731 0 -1 0 "
            "0 0 0 0 0 0 0 0 20 0 2 0 100\n"
        )

        def list_directory(path):
            if path == "/proc":
                return ["731"]
            if path == "/proc/731/fd":
                return []
            if path == "/proc/731/task":
                return ["731"]
            raise AssertionError("unexpected listdir %s" % path)

        def open_proc(path, *args, **kwargs):
            del args, kwargs
            if path == "/proc/731/stat":
                return io.StringIO(live_stat)
            if path == "/proc/731/maps":
                return io.StringIO("")
            raise AssertionError("unexpected open %s" % path)

        with mock.patch.object(
            linux_ops, "_require_linux"
        ), mock.patch.object(
            linux_ops.os, "listdir", side_effect=list_directory
        ), mock.patch.object(
            linux_ops.os, "readlink", return_value="/"
        ), mock.patch(
            "builtins.open", side_effect=open_proc
        ):
            with self.assertRaisesRegex(
                ramdisk.RamdiskError,
                "incomplete process task snapshot",
            ):
                linux_ops._busy_mount_references_proc("/mnt/colibri")

    def test_busy_mount_scan_rejects_live_nonleader_endpoint_disappearance(self):
        live_stat = (
            "731 (worker) S 1 731 731 0 -1 0 "
            "0 0 0 0 0 0 0 0 20 0 2 0 100\n"
        )
        task_stat = live_stat.replace("731 (worker)", "732 (worker)", 1)

        def list_directory(path):
            if path == "/proc":
                return ["731"]
            if path == "/proc/731/fd":
                return []
            if path == "/proc/731/task":
                return ["731", "732"]
            raise AssertionError("unexpected listdir %s" % path)

        def read_link(path):
            if path == "/proc/731/task/732/cwd":
                raise FileNotFoundError(errno.ENOENT, "task endpoint vanished", path)
            return "/"

        def open_proc(path, *args, **kwargs):
            del args, kwargs
            if path == "/proc/731/stat":
                return io.StringIO(live_stat)
            if path == "/proc/731/maps":
                return io.StringIO("")
            if path == "/proc/731/task/732/stat":
                return io.StringIO(task_stat)
            raise AssertionError("unexpected open %s" % path)

        with mock.patch.object(
            linux_ops, "_require_linux"
        ), mock.patch.object(
            linux_ops.os, "listdir", side_effect=list_directory
        ), mock.patch.object(
            linux_ops.os, "readlink", side_effect=read_link
        ), mock.patch(
            "builtins.open", side_effect=open_proc
        ):
            with self.assertRaisesRegex(
                ramdisk.RamdiskError,
                "missing task endpoint .* task 732 .* remains live",
            ):
                linux_ops._busy_mount_references_proc("/mnt/colibri")

    def test_busy_mount_scan_preserves_non_utf8_leader_and_task_names(self):
        leader_stat = (
            b"731 (leader-\xff) S 1 731 731 0 -1 0 "
            b"0 0 0 0 0 0 0 0 20 0 2 0 100\n"
        )
        task_stat = (
            b"732 (task-\xfe) Z 1 731 731 0 -1 0 "
            b"0 0 0 0 0 0 0 0 20 0 1 0 101\n"
        )
        stat_reads = []

        def list_directory(path):
            if path == "/proc":
                return ["731"]
            if path == "/proc/731/fd":
                return []
            if path == "/proc/731/task":
                return ["731", "732"]
            raise AssertionError("unexpected listdir %s" % path)

        def read_link(path):
            if path == "/proc/731/task/732/cwd":
                raise FileNotFoundError(
                    errno.ENOENT,
                    "zombie task has no cwd",
                    path,
                )
            return "/"

        def open_proc(path, mode, *, encoding, errors, newline=None):
            self.assertEqual(mode, "r")
            if path == "/proc/731/maps":
                self.assertIsNone(newline)
                return io.StringIO("")
            self.assertEqual(newline, "")
            if path == "/proc/731/stat":
                stat_reads.append((path, errors))
                payload = leader_stat
            elif path == "/proc/731/task/732/stat":
                stat_reads.append((path, errors))
                payload = task_stat
            else:
                raise AssertionError("unexpected open %s" % path)
            return io.TextIOWrapper(
                io.BytesIO(payload),
                encoding=encoding,
                errors=errors,
                newline=newline,
            )

        with mock.patch.object(
            linux_ops, "_require_linux"
        ), mock.patch.object(
            linux_ops.os, "listdir", side_effect=list_directory
        ), mock.patch.object(
            linux_ops.os, "readlink", side_effect=read_link
        ), mock.patch(
            "builtins.open", side_effect=open_proc
        ):
            self.assertEqual(
                linux_ops._busy_mount_references_proc("/mnt/colibri"),
                [],
            )

        self.assertEqual(
            stat_reads,
            [
                ("/proc/731/stat", "surrogateescape"),
                ("/proc/731/task/732/stat", "surrogateescape"),
                ("/proc/731/stat", "surrogateescape"),
            ],
        )

    def test_unprivileged_busy_scan_uses_trusted_fuser_without_shell(self):
        trusted = mock.Mock(
            side_effect=lambda name: "/usr/bin/" + name
        )
        run = mock.Mock(
            return_value=subprocess.CompletedProcess(
                [],
                0,
                " 732 731 732\n",
                "/mnt/colibri: mmm\n",
            )
        )
        with mock.patch.object(
            linux_ops, "_require_linux"
        ), mock.patch.object(
            linux_ops, "current_euid", return_value=1000
        ):
            result = linux_ops._busy_mount_references(
                "/mnt/colibri",
                hardware=mock.sentinel.hardware,
                run=run,
                trusted_system_binary=trusted,
            )

        self.assertEqual(result, [731, 732])
        run.assert_called_once_with(
            [
                "/usr/bin/sudo",
                "--",
                "/usr/bin/fuser",
                "-mM",
                "/mnt/colibri",
            ],
            timeout=10.0,
        )
        self.assertEqual(
            trusted.call_args_list,
            [mock.call("fuser"), mock.call("sudo")],
        )

    def test_fuser_parser_distinguishes_empty_from_diagnostics(self):
        empty = subprocess.CompletedProcess([], 1, "", "")
        self.assertEqual(
            linux_ops._parse_fuser_mount_references(
                "/mnt/colibri",
                empty,
            ),
            [],
        )
        self.assertEqual(
            linux_ops._parse_fuser_mount_references(
                "/mnt/coli bri",
                subprocess.CompletedProcess(
                    [],
                    0,
                    "731 732\n",
                    "/mnt/coli bri: rcemF\n",
                ),
            ),
            [731, 732],
        )

        failures = (
            subprocess.CompletedProcess(
                [],
                1,
                "",
                "Specified filename is not a mountpoint",
            ),
            subprocess.CompletedProcess([], 2, "", "fatal"),
            subprocess.CompletedProcess([], 0, "", "annotations"),
            subprocess.CompletedProcess([], 0, "731 nope", "annotations"),
            subprocess.CompletedProcess([], 0, "0", "annotations"),
            subprocess.CompletedProcess(
                [],
                0,
                "731",
                "/mnt/colibri: m\nCannot stat file: Permission denied\n",
            ),
            subprocess.CompletedProcess(
                [],
                0,
                "731",
                "/mnt/colibri: m\nCannot open a network socket.\n",
            ),
        )
        for result in failures:
            with self.subTest(
                returncode=result.returncode,
                stdout=result.stdout,
            ):
                with self.assertRaises(ramdisk.RamdiskError):
                    linux_ops._parse_fuser_mount_references(
                        "/mnt/colibri",
                        result,
                    )

    def test_unprivileged_busy_scan_fails_closed_on_runner_exception(self):
        with mock.patch.object(
            linux_ops, "_require_linux"
        ), mock.patch.object(
            linux_ops, "current_euid", return_value=1000
        ):
            with self.assertRaisesRegex(
                ramdisk.RamdiskError,
                "trusted fuser could not inspect managed mount",
            ):
                linux_ops._busy_mount_references(
                    "/mnt/colibri",
                    run=mock.Mock(
                        side_effect=subprocess.TimeoutExpired("fuser", 10)
                    ),
                    trusted_system_binary=(
                        lambda name: "/usr/bin/" + name
                    ),
                )

    def test_prepare_cleanup_capability_resolves_fuser_sudo_and_umount(self):
        trusted = mock.Mock(
            side_effect=lambda name: "/usr/bin/" + name
        )

        def run_helper(command, **kwargs):
            del kwargs
            if command[-3:] == ["/usr/bin/fuser", "-mM", "/"]:
                return subprocess.CompletedProcess(
                    command,
                    0,
                    "731\n",
                    "/: m\n",
                )
            if command[-2:] == ["/usr/bin/umount", "--help"]:
                return subprocess.CompletedProcess(
                    command,
                    0,
                    "  -c, --no-canonicalize\n",
                    "",
                )
            raise AssertionError("unexpected command %r" % (command,))

        run = mock.Mock(
            side_effect=run_helper
        )
        with mock.patch.object(
            linux_ops, "_require_linux"
        ), mock.patch.object(
            linux_ops, "current_euid", return_value=1000
        ), linux_ops._noninteractive_privilege(
            trusted_system_binary=trusted,
        ):
            linux_ops._ensure_busy_mount_scan_available(
                "/mnt/colibri",
                trusted_system_binary=trusted,
                run=run,
            )

        self.assertEqual(
            trusted.call_args_list,
            [
                mock.call("fuser"),
                mock.call("sudo"),
                mock.call("umount"),
                mock.call("sudo"),
            ],
        )
        self.assertEqual(
            run.call_args_list,
            [
                mock.call(
                    [
                        "/usr/bin/sudo",
                        "-n",
                        "--",
                        "/usr/bin/fuser",
                        "-mM",
                        "/",
                    ],
                    timeout=10.0,
                ),
                mock.call(
                    [
                        "/usr/bin/sudo",
                        "-n",
                        "--",
                        "/usr/bin/umount",
                        "--help",
                    ],
                    timeout=5.0,
                ),
            ],
        )

    def test_prepare_rejects_denied_privileged_umount_probe(self):
        trusted = mock.Mock(
            side_effect=lambda name: "/usr/bin/" + name
        )

        def run_helper(command, **kwargs):
            del kwargs
            if command[-3:] == ["/usr/bin/fuser", "-mM", "/"]:
                return subprocess.CompletedProcess(
                    command,
                    0,
                    "731\n",
                    "/: m\n",
                )
            return subprocess.CompletedProcess(
                command,
                1,
                "",
                "sudo: command not allowed\n",
            )

        run = mock.Mock(side_effect=run_helper)
        with mock.patch.object(
            linux_ops, "_require_linux"
        ), mock.patch.object(
            linux_ops, "current_euid", return_value=1000
        ):
            with self.assertRaisesRegex(
                ramdisk.RamdiskError,
                "umount helper is incompatible or unauthorized",
            ):
                linux_ops._ensure_busy_mount_scan_available(
                    "/mnt/colibri",
                    trusted_system_binary=trusted,
                    run=run,
                )

        self.assertEqual(
            run.call_args_list[-1],
            mock.call(
                [
                    "/usr/bin/sudo",
                    "--",
                    "/usr/bin/umount",
                    "--help",
                ],
                timeout=5.0,
            ),
        )

    def test_prepare_rejects_denied_fuser_probe_before_umount_check(self):
        trusted = mock.Mock(
            side_effect=lambda name: "/usr/bin/" + name
        )
        denied = subprocess.CompletedProcess(
            [],
            1,
            "",
            "sudo: a password is required\n",
        )
        run = mock.Mock(return_value=denied)

        with mock.patch.object(
            linux_ops, "_require_linux"
        ), mock.patch.object(
            linux_ops, "current_euid", return_value=1000
        ):
            with self.assertRaisesRegex(
                ramdisk.RamdiskError,
                "trusted fuser could not inspect managed mount /",
            ):
                linux_ops._ensure_busy_mount_scan_available(
                    "/mnt/colibri",
                    trusted_system_binary=trusted,
                    run=run,
                )

        self.assertEqual(
            trusted.call_args_list,
            [mock.call("fuser"), mock.call("sudo")],
        )
        run.assert_called_once_with(
            [
                "/usr/bin/sudo",
                "--",
                "/usr/bin/fuser",
                "-mM",
                "/",
            ],
            timeout=10.0,
        )

    def test_root_prepare_cleanup_capability_resolves_umount(self):
        trusted = mock.Mock(
            side_effect=lambda name: "/usr/bin/" + name
        )
        run = mock.Mock(
            return_value=subprocess.CompletedProcess(
                [], 0, "--no-canonicalize", ""
            )
        )
        with mock.patch.object(
            linux_ops, "_require_linux"
        ), mock.patch.object(
            linux_ops, "current_euid", return_value=0
        ), mock.patch.object(
            linux_ops, "_busy_mount_references_proc", return_value=[]
        ) as scan:
            linux_ops._ensure_busy_mount_scan_available(
                "/mnt/colibri",
                trusted_system_binary=trusted,
                run=run,
            )

        scan.assert_called_once_with("/mnt/colibri")
        trusted.assert_called_once_with("umount")
        run.assert_called_once_with(["/usr/bin/umount", "--help"], timeout=5.0)

    def test_prepare_rejects_incompatible_umount_before_mount_mutation(self):
        run = mock.Mock(
            return_value=subprocess.CompletedProcess(
                [], 0, "usage: umount TARGET", ""
            )
        )
        with mock.patch.object(
            linux_ops, "_require_linux"
        ), mock.patch.object(
            linux_ops, "current_euid", return_value=0
        ), mock.patch.object(
            linux_ops, "_busy_mount_references_proc", return_value=[]
        ):
            with self.assertRaisesRegex(
                ramdisk.RamdiskError,
                "util-linux --no-canonicalize",
            ):
                linux_ops._ensure_busy_mount_scan_available(
                    "/mnt/colibri",
                    trusted_system_binary=lambda name: "/trusted/" + name,
                    run=run,
                )

        run.assert_called_once_with(
            ["/trusted/umount", "--help"],
            timeout=5.0,
        )

    def test_missing_fuser_names_the_required_psmisc_package(self):
        trusted = mock.Mock(
            side_effect=ramdisk.RamdiskError(
                "trusted fuser executable was not found"
            )
        )
        with mock.patch.object(
            linux_ops, "_require_linux"
        ), mock.patch.object(
            linux_ops, "current_euid", return_value=1000
        ):
            with self.assertRaisesRegex(
                ramdisk.RamdiskError,
                "install the psmisc package",
            ):
                linux_ops._ensure_busy_mount_scan_available(
                    "/mnt/colibri",
                    trusted_system_binary=trusted,
                )

        trusted.assert_called_once_with("fuser")


if __name__ == "__main__":
    unittest.main()
