import argparse
import errno
import io
import json
import os
import signal
import stat
import subprocess
import sys
import tempfile
import time
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
        requires_linux_pidfd,
    )
else:
    from platform_test_support import (  # noqa: E402
        PLATFORM_SKIP_INVENTORY,
        assert_platform_skip_inventory,
        requires_linux_pidfd,
    )


def _proc_stat_record(pid, pgid, session, starttime, state="S"):
    fields = [state, "1", str(pgid), str(session)] + ["0"] * 15
    fields[17] = "1"
    fields.append(str(starttime))
    return "%d (managed worker) %s\n" % (pid, " ".join(fields))


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

    def test_linux_process_mutator_reports_missing_pidfd_before_state(self):
        ops = linux_ops.LinuxPlatformOps()
        forbidden_lock = mock.Mock(
            side_effect=AssertionError("lifecycle lock was reached")
        )
        with mock.patch.object(
            linux_ops,
            "_pidfd_process_control_supported",
            return_value=False,
        ), mock.patch.object(
            ramdisk,
            "get_platform_ops",
            return_value=ops,
        ), mock.patch.object(
            ramdisk,
            "_lifecycle_lock",
            forbidden_lock,
        ):
            with self.assertRaisesRegex(
                ramdisk.RamdiskError,
                "requires Linux pidfd_open.*pidfd_send_signal",
            ):
                ramdisk.stop()

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

    def test_managed_launch_discovery_is_explicitly_unsupported_off_linux(self):
        ops = UnsupportedPlatformOps("win32")
        with self.assertRaisesRegex(
            ramdisk.RamdiskError,
            UNSUPPORTED_PLATFORM_REASON,
        ):
            ops.process_start_boundary()
        with self.assertRaisesRegex(
            ramdisk.RamdiskError,
            UNSUPPORTED_PLATFORM_REASON,
        ):
            ops.managed_launch_processes(
                nonce="a" * 48,
                uid=1000,
                state_dir="/state/node-0",
                weights_dir="/mnt/weights",
                not_before_starttime=17000,
                launcher_pid=700,
                launcher_starttime=16000,
                launcher_cmdline=["coli", "ramdisk", "start"],
                            expected_command=["coli", "serve"],
            )

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
            38,
        )
        self.assertEqual(
            len(PLATFORM_SKIP_INVENTORY["sigterm_handler"]["tests"]),
            6,
        )
        self.assertEqual(
            len(PLATFORM_SKIP_INVENTORY["sigint_handler"]["tests"]),
            4,
        )
        self.assertEqual(
            len(PLATFORM_SKIP_INVENTORY["posix_pty"]["tests"]),
            1,
        )
        self.assertEqual(
            len(PLATFORM_SKIP_INVENTORY["posix_fifo"]["tests"]),
            1,
        )
        self.assertEqual(
            len(PLATFORM_SKIP_INVENTORY["native_dirfd"]["tests"]),
            4,
        )
        self.assertEqual(
            len(PLATFORM_SKIP_INVENTORY["linux_pidfd"]["tests"]),
            1,
        )
        self.assertEqual(
            len(PLATFORM_SKIP_INVENTORY["linux_stdlib_pidfd"]["tests"]),
            1,
        )
        assert_platform_skip_inventory()

    def test_pidfd_markers_distinguish_managed_libc_from_stdlib_support(self):
        target = os.environ.get("COLIBRI_TEST_TARGET_PLATFORM", sys.platform)
        managed_pidfd = (
            target.startswith("linux")
            and linux_ops._pidfd_process_control_supported()
        )

        self.assertEqual(
            PLATFORM_SKIP_INVENTORY["linux_pidfd"]["supported"],
            managed_pidfd,
        )
        self.assertEqual(
            PLATFORM_SKIP_INVENTORY["linux_stdlib_pidfd"]["supported"],
            target.startswith("linux")
            and callable(getattr(os, "pidfd_open", None))
            and callable(getattr(signal, "pidfd_send_signal", None)),
        )


class LinuxOperationalReadContractTest(unittest.TestCase):
    @requires_linux_pidfd
    def test_real_pidfd_group_signal_targets_each_exact_member(self):
        nonce = "d" * 48
        state_dir = "/tmp/colibri-pidfd-test-state"
        weights_dir = "/tmp/colibri-pidfd-test-weights"
        environment = os.environ.copy()
        environment.update(
            COLI_MANAGED_NONCE=nonce,
            COLI_STATE_DIR=state_dir,
            COLI_WEIGHTS_DIR=weights_dir,
        )
        program = (
            "import os,time; child=os.fork(); "
            "print(child,flush=True) if child else None; time.sleep(60)"
        )
        process = subprocess.Popen(
            [sys.executable, "-c", program],
            start_new_session=True,
            stdout=subprocess.PIPE,
            text=True,
            env=environment,
        )
        child_pid = int(process.stdout.readline().strip())
        open_pidfd, send_pidfd = linux_ops._pidfd_api()
        cleanup_pidfds = []
        for pid in (process.pid, child_pid):
            cleanup_pidfds.append(open_pidfd(pid, 0))
        real_killpg = linux_ops.os.killpg

        try:
            leader = linux_ops._strict_process_identity(process.pid)
            record = {
                "pid": process.pid,
                "pgid": process.pid,
                "uid": os.getuid(),
                "starttime": leader["starttime"],
                "nonce": nonce,
                "state_dir": state_dir,
                "weights_dir": weights_dir,
            }
            with mock.patch.object(
                linux_ops.os,
                "killpg",
                side_effect=lambda pgid, signum: real_killpg(pgid, signum),
            ) as killpg:
                result = linux_ops._signal_verified_process_group(
                    record,
                    signal.SIGTERM,
                )
                process.wait(timeout=5.0)
                deadline = time.monotonic() + 5.0
                while (
                    time.monotonic() < deadline
                    and linux_ops._process_group_alive(process.pid)
                ):
                    time.sleep(0.05)

            self.assertEqual(result["status"], "signaled")
            self.assertEqual(
                set(result["signaled"]),
                {process.pid, child_pid},
            )
            self.assertEqual(process.returncode, -signal.SIGTERM)
            self.assertFalse(linux_ops._process_group_alive(process.pid))
            self.assertTrue(
                all(call.args[1] == 0 for call in killpg.call_args_list)
            )
        finally:
            for descriptor in cleanup_pidfds:
                try:
                    send_pidfd(descriptor, signal.SIGKILL, None, 0)
                except OSError:
                    pass
                os.close(descriptor)
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5.0)
            process.stdout.close()

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

    def test_process_start_boundary_floors_uptime_to_boot_ticks(self):
        with mock.patch.object(
            linux_ops, "_require_linux"
        ), mock.patch.object(
            linux_ops.os, "sysconf", return_value=250
        ), mock.patch(
            "builtins.open",
            return_value=io.StringIO("123.456 99.0\n"),
        ):
            boundary = linux_ops.LinuxPlatformOps().process_start_boundary()

        self.assertEqual(boundary, 30864)

    def test_process_start_boundary_fails_closed(self):
        cases = (
            ("unreadable", 100, None, "cannot read Linux boot uptime"),
            ("malformed", 100, "unknown 0.0\n", "cannot parse Linux boot uptime"),
            ("invalid-hz", 0, "123.45 0.0\n", "clock tick rate is invalid"),
        )
        for case, ticks_per_second, uptime, message in cases:
            def open_uptime(*args, **kwargs):
                del args, kwargs
                if uptime is None:
                    raise PermissionError(
                        errno.EACCES,
                        "permission denied",
                        "/proc/uptime",
                    )
                return io.StringIO(uptime)

            with self.subTest(case=case), mock.patch.object(
                linux_ops, "_require_linux"
            ), mock.patch.object(
                linux_ops.os,
                "sysconf",
                return_value=ticks_per_second,
            ), mock.patch(
                "builtins.open", side_effect=open_uptime
            ):
                with self.assertRaisesRegex(ramdisk.RamdiskError, message):
                    linux_ops._process_start_boundary()

    def test_managed_launch_scan_returns_every_exact_attributed_identity(self):
        nonce = "a" * 48
        owners = {731: 1000, 732: 1000, 900: 2000}
        starttimes = {731: 17001, 732: 17002}
        opened = []

        def file_stat(path):
            pid = int(path.rsplit("/", 1)[-1])
            return mock.Mock(st_uid=owners[pid])

        def open_proc(path, mode, *args, **kwargs):
            del args, kwargs
            opened.append(path)
            pid = int(path.split("/")[2])
            if path.endswith("/stat"):
                self.assertEqual(mode, "r")
                return io.StringIO(
                    _proc_stat_record(pid, 731, 731, starttimes[pid])
                )
            if path.endswith("/cmdline"):
                self.assertEqual(mode, "rb")
                return io.BytesIO(
                    ("coli\0serve\0--port\0%d\0" % (8000 + pid - 731)).encode()
                )
            if path.endswith("/environ"):
                self.assertEqual(mode, "rb")
                return io.BytesIO(
                    (
                        "COLI_MANAGED_NONCE=%s\0"
                        "COLI_STATE_DIR=/state/node-0\0"
                        "COLI_WEIGHTS_DIR=/mnt/weights\0" % nonce
                    ).encode()
                )
            raise AssertionError("unexpected open %s" % path)

        with mock.patch.object(
            linux_ops, "_require_linux"
        ), mock.patch.object(
            linux_ops.os,
            "listdir",
            return_value=["732", "self", "900", "731"],
        ), mock.patch.object(
            linux_ops.os, "stat", side_effect=file_stat
        ), mock.patch(
            "builtins.open", side_effect=open_proc
        ):
            result = linux_ops.LinuxPlatformOps().managed_launch_processes(
                nonce=nonce,
                uid=1000,
                state_dir="/state/node-0",
                weights_dir="/mnt/weights",
                not_before_starttime=17000,
                launcher_pid=700,
                launcher_starttime=16000,
                launcher_cmdline=["coli", "ramdisk", "start"],
                            expected_command=["coli", "serve"],
            )

        self.assertEqual([item["pid"] for item in result], [731, 732])
        self.assertEqual(result[0]["pgid"], 731)
        self.assertEqual(result[0]["sid"], 731)
        self.assertEqual(result[0]["starttime"], 17001)
        self.assertEqual(result[0]["nonce"], nonce)
        self.assertEqual(result[0]["cmdline"][:2], ["coli", "serve"])
        self.assertEqual(result[0]["state_dir"], "/state/node-0")
        self.assertEqual(result[0]["weights_dir"], "/mnt/weights")
        self.assertFalse(any(path.startswith("/proc/900/") for path in opened))

    def test_process_identity_exposes_group_attribution_fields(self):
        nonce = "a" * 48

        def open_proc(path, mode, *args, **kwargs):
            del args, kwargs
            if path == "/proc/732/stat":
                self.assertEqual(mode, "r")
                return io.StringIO(
                    _proc_stat_record(732, 731, 731, 17002)
                )
            if path == "/proc/732/cmdline":
                self.assertEqual(mode, "rb")
                return io.BytesIO(b"coli\0serve\0")
            if path == "/proc/732/environ":
                self.assertEqual(mode, "rb")
                return io.BytesIO(
                    (
                        "COLI_MANAGED_NONCE=%s\0"
                        "COLI_STATE_DIR=/state/node-0\0"
                        "COLI_WEIGHTS_DIR=/mnt/weights\0" % nonce
                    ).encode()
                )
            raise AssertionError("unexpected open %s" % path)

        with mock.patch.object(
            linux_ops, "_require_linux"
        ), mock.patch.object(
            linux_ops.os, "getpgid", return_value=731
        ), mock.patch.object(
            linux_ops.os,
            "stat",
            return_value=mock.Mock(st_uid=1000),
        ), mock.patch(
            "builtins.open", side_effect=open_proc
        ):
            identity = linux_ops._process_identity(732)

        self.assertEqual(identity["pid"], 732)
        self.assertEqual(identity["uid"], 1000)
        self.assertEqual(identity["starttime"], 17002)
        self.assertEqual(identity["pgid"], 731)
        self.assertEqual(identity["sid"], 731)
        self.assertEqual(identity["nonce"], nonce)
        self.assertEqual(identity["state_dir"], "/state/node-0")
        self.assertEqual(identity["weights_dir"], "/mnt/weights")

    def test_process_identity_preserves_none_for_unreadable_attribution(self):
        def open_proc(path, mode, *args, **kwargs):
            del mode, args, kwargs
            if path.endswith("/stat"):
                return io.StringIO(
                    _proc_stat_record(732, 731, 731, 17002)
                )
            if path.endswith("/cmdline"):
                return io.BytesIO(b"coli\0serve\0")
            if path.endswith("/environ"):
                raise PermissionError(
                    errno.EACCES,
                    "permission denied",
                    path,
                )
            raise AssertionError("unexpected open %s" % path)

        with mock.patch.object(
            linux_ops, "_require_linux"
        ), mock.patch.object(
            linux_ops.os, "getpgid", return_value=731
        ), mock.patch(
            "builtins.open", side_effect=open_proc
        ):
            self.assertIsNone(linux_ops._process_identity(732))

    def test_process_identity_rejects_a_hybrid_reused_pid_snapshot(self):
        before = {
            "state": "S",
            "starttime": 17002,
            "pgid": 731,
            "sid": 731,
            "num_threads": 1,
        }
        after = dict(before, starttime=17003)

        def open_proc(path, mode, *args, **kwargs):
            del args, kwargs
            if path.endswith("/cmdline"):
                self.assertEqual(mode, "rb")
                return io.BytesIO(b"coli\0serve\0")
            if path.endswith("/environ"):
                self.assertEqual(mode, "rb")
                return io.BytesIO(b"OTHER=value\0")
            raise AssertionError("unexpected open %s" % path)

        with mock.patch.object(
            linux_ops, "_require_linux"
        ), mock.patch.object(
            linux_ops,
            "_strict_proc_stat_identity",
            side_effect=[before, after],
        ), mock.patch.object(
            linux_ops.os, "getpgid", return_value=731
        ), mock.patch.object(
            linux_ops.os,
            "stat",
            return_value=mock.Mock(st_uid=1000),
        ), mock.patch("builtins.open", side_effect=open_proc):
            self.assertIsNone(linux_ops._process_identity(732))

    def test_process_identity_proves_a_lone_thread_zombie_inert(self):
        opened = []

        def open_proc(path, mode, *args, **kwargs):
            del args, kwargs
            opened.append(path)
            if path.endswith("/stat"):
                self.assertEqual(mode, "r")
                return io.StringIO(
                    _proc_stat_record(732, 731, 731, 17002, state="Z")
                )
            raise AssertionError("zombie endpoint must not be read: %s" % path)

        with mock.patch.object(
            linux_ops, "_require_linux"
        ), mock.patch.object(
            linux_ops.os,
            "stat",
            return_value=mock.Mock(st_uid=1000),
        ), mock.patch.object(
            linux_ops.os, "listdir", return_value=["732"]
        ), mock.patch("builtins.open", side_effect=open_proc):
            identity = linux_ops._process_identity(732)

        self.assertTrue(identity["inert"])
        self.assertEqual(identity["state"], "Z")
        self.assertEqual(identity["starttime"], 17002)
        self.assertNotIn("/proc/732/cmdline", opened)
        self.assertNotIn("/proc/732/environ", opened)

    def test_managed_launch_inspection_treats_a_stable_zombie_as_inert(self):
        opened = []

        def open_proc(path, mode, *args, **kwargs):
            del args, kwargs
            opened.append(path)
            if path.endswith("/stat"):
                self.assertEqual(mode, "r")
                return io.StringIO(
                    _proc_stat_record(731, 731, 731, 17000, state="Z")
                )
            raise AssertionError("zombie endpoint must not be read: %s" % path)

        with mock.patch.object(
            linux_ops, "_require_linux"
        ), mock.patch.object(
            linux_ops.os,
            "stat",
            return_value=mock.Mock(st_uid=1000),
        ), mock.patch.object(
            linux_ops.os, "listdir", return_value=["731"]
        ), mock.patch("builtins.open", side_effect=open_proc):
            observation = linux_ops._inspect_managed_launch_pid(
                731,
                1000,
                17000,
            )

        self.assertEqual(observation["kind"], "inert-dead")
        self.assertEqual(observation["state"], "Z")
        self.assertNotIn("/proc/731/cmdline", opened)
        self.assertNotIn("/proc/731/environ", opened)

    def test_inert_identity_rejects_incomplete_or_changing_task_snapshot(self):
        before = {
            "state": "Z",
            "starttime": 17000,
            "pgid": 731,
            "sid": 731,
            "num_threads": 1,
        }
        with self.subTest(case="unexpected-sibling"), mock.patch.object(
            linux_ops.os,
            "listdir",
            return_value=["731", "732"],
        ):
            with self.assertRaisesRegex(
                ramdisk.RamdiskError,
                "incomplete or live task group",
            ):
                linux_ops._stable_inert_process_identity(
                    731,
                    "/proc/731",
                    1000,
                    before,
                )

        changed = dict(before, num_threads=2)
        with self.subTest(case="stat-turnover"), mock.patch.object(
            linux_ops.os,
            "listdir",
            return_value=["731"],
        ), mock.patch.object(
            linux_ops,
            "_strict_proc_stat_identity",
            return_value=changed,
        ), mock.patch.object(
            linux_ops.os,
            "stat",
            return_value=mock.Mock(st_uid=1000),
        ):
            with self.assertRaisesRegex(
                ramdisk.RamdiskError,
                "changed during pending-launch recovery",
            ):
                linux_ops._stable_inert_process_identity(
                    731,
                    "/proc/731",
                    1000,
                    before,
                )

    def test_managed_launch_scan_skips_only_old_unreadable_same_uid(self):
        for starttime, must_refuse in ((16999, False), (17000, True)):
            opened = []

            def open_proc(path, mode, *args, **kwargs):
                del mode, args, kwargs
                opened.append(path)
                if path.endswith("/stat"):
                    return io.StringIO(
                        _proc_stat_record(731, 731, 731, starttime)
                    )
                if path.endswith("/cmdline"):
                    return io.BytesIO(b"unrelated\0")
                if path.endswith("/environ"):
                    raise PermissionError(
                        errno.EACCES,
                        "permission denied",
                        path,
                    )
                raise AssertionError("unexpected open %s" % path)

            with self.subTest(
                starttime=starttime
            ), mock.patch.object(
                linux_ops, "_require_linux"
            ), mock.patch.object(
                linux_ops.os, "listdir", return_value=["731"]
            ), mock.patch.object(
                linux_ops.os,
                "stat",
                return_value=mock.Mock(st_uid=1000),
            ), mock.patch(
                "builtins.open", side_effect=open_proc
            ):
                if must_refuse:
                    with self.assertRaisesRegex(
                        ramdisk.RamdiskError,
                        "cannot read Linux process identity /proc/731/environ",
                    ):
                        linux_ops._managed_launch_processes(
                            "a" * 48,
                            1000,
                            state_dir="/state/node-0",
                            weights_dir="/mnt/weights",
                            not_before_starttime=17000,
                            launcher_pid=700,
                            launcher_starttime=16000,
                            launcher_cmdline=["coli", "ramdisk", "start"],
                            expected_command=["coli", "serve"],
                        )
                    self.assertIn("/proc/731/environ", opened)
                else:
                    self.assertEqual(
                        linux_ops._managed_launch_processes(
                            "a" * 48,
                            1000,
                            state_dir="/state/node-0",
                            weights_dir="/mnt/weights",
                            not_before_starttime=17000,
                            launcher_pid=700,
                            launcher_starttime=16000,
                            launcher_cmdline=["coli", "ramdisk", "start"],
                            expected_command=["coli", "serve"],
                        ),
                        [],
                    )
                    self.assertNotIn("/proc/731/cmdline", opened)
                    self.assertNotIn("/proc/731/environ", opened)

    def test_managed_launch_scan_rejects_recent_missing_or_wrong_nonce(self):
        nonce = "a" * 48
        for case, actual_nonce in (("missing", None), ("wrong", "b" * 48)):
            def open_proc(path, mode, *args, **kwargs):
                del mode, args, kwargs
                if path.endswith("/stat"):
                    return io.StringIO(
                        _proc_stat_record(731, 731, 731, 17000)
                    )
                if path.endswith("/cmdline"):
                    return io.BytesIO(b"coli\0ramdisk\0start\0")
                if path.endswith("/environ"):
                    nonce_field = (
                        "COLI_MANAGED_NONCE=%s\0" % actual_nonce
                        if actual_nonce is not None
                        else ""
                    )
                    return io.BytesIO(
                        (
                            nonce_field
                            + "COLI_STATE_DIR=/state/node-0\0"
                            + "COLI_WEIGHTS_DIR=/mnt/weights\0"
                        ).encode()
                    )
                raise AssertionError("unexpected open %s" % path)

            with self.subTest(case=case), mock.patch.object(
                linux_ops, "_require_linux"
            ), mock.patch.object(
                linux_ops.os, "listdir", return_value=["731"]
            ), mock.patch.object(
                linux_ops.os,
                "stat",
                return_value=mock.Mock(st_uid=1000),
            ), mock.patch(
                "builtins.open", side_effect=open_proc
            ):
                with self.assertRaisesRegex(
                    ramdisk.RamdiskError,
                    "(ambiguous managed launch attribution|"
                    "missing or mismatched nonce attribution)",
                ):
                    linux_ops._managed_launch_processes(
                        nonce,
                        1000,
                        state_dir="/state/node-0",
                        weights_dir="/mnt/weights",
                        not_before_starttime=17000,
                        launcher_pid=700,
                        launcher_starttime=16000,
                        launcher_cmdline=["coli", "ramdisk", "start"],
                            expected_command=["coli", "serve"],
                    )

    def test_managed_launch_scan_ignores_stable_readable_recent_bystander(self):
        def open_proc(path, mode, *args, **kwargs):
            del mode, args, kwargs
            if path.endswith("/stat"):
                return io.StringIO(
                    _proc_stat_record(731, 731, 731, 17000)
                )
            if path.endswith("/cmdline"):
                return io.BytesIO(b"unrelated-worker\0")
            if path.endswith("/environ"):
                return io.BytesIO(
                    b"BROKEN\0"
                    b"COLI_MANAGED_NONCE=bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\0"
                    b"COLI_MANAGED_NONCE\0"
                    b"COLI_STATE_DIR=/other/a\0"
                    b"COLI_STATE_DIR=/other/b\0"
                    b"COLI_WEIGHTS_DIR=/other/weights\0"
                )
            raise AssertionError("unexpected open %s" % path)

        with mock.patch.object(
            linux_ops, "_require_linux"
        ), mock.patch.object(
            linux_ops.os, "listdir", return_value=["731"]
        ), mock.patch.object(
            linux_ops.os,
            "stat",
            return_value=mock.Mock(st_uid=1000),
        ), mock.patch("builtins.open", side_effect=open_proc):
            self.assertEqual(
                linux_ops._managed_launch_processes(
                    "a" * 48,
                    1000,
                    state_dir="/state/node-0",
                    weights_dir="/mnt/weights",
                    not_before_starttime=17000,
                    launcher_pid=700,
                    launcher_starttime=16000,
                    launcher_cmdline=["coli", "ramdisk", "start"],
                    expected_command=["coli", "serve"],
                ),
                [],
            )

    def test_managed_launch_scan_rejects_path_attribution_without_target_nonce(self):
        for case, actual_nonce in (("missing", None), ("wrong", "b" * 48)):
            def open_proc(path, mode, *args, **kwargs):
                del mode, args, kwargs
                if path.endswith("/stat"):
                    return io.StringIO(
                        _proc_stat_record(731, 731, 731, 17000)
                    )
                if path.endswith("/cmdline"):
                    return io.BytesIO(b"renamed-engine\0")
                if path.endswith("/environ"):
                    nonce_field = (
                        ("COLI_MANAGED_NONCE=%s\0" % actual_nonce).encode()
                        if actual_nonce is not None
                        else b""
                    )
                    return io.BytesIO(
                        nonce_field
                        + b"COLI_STATE_DIR=/state/node-0\0"
                        + b"COLI_WEIGHTS_DIR=/mnt/weights\0"
                    )
                raise AssertionError("unexpected open %s" % path)

            with self.subTest(case=case), mock.patch.object(
                linux_ops, "_require_linux"
            ), mock.patch.object(
                linux_ops.os, "listdir", return_value=["731"]
            ), mock.patch.object(
                linux_ops.os,
                "stat",
                return_value=mock.Mock(st_uid=1000),
            ), mock.patch("builtins.open", side_effect=open_proc):
                with self.assertRaisesRegex(
                    ramdisk.RamdiskError,
                    "(ambiguous managed launch attribution|"
                    "missing or mismatched nonce attribution)",
                ):
                    linux_ops._managed_launch_processes(
                        "a" * 48,
                        1000,
                        state_dir="/state/node-0",
                        weights_dir="/mnt/weights",
                        not_before_starttime=17000,
                        launcher_pid=700,
                        launcher_starttime=16000,
                        launcher_cmdline=["coli", "ramdisk", "start"],
                        expected_command=["coli", "serve"],
                    )

    def test_managed_launch_scan_excludes_the_exact_original_launcher(self):
        observation = {
            "kind": "same-uid",
            "pid": 731,
            "uid": 1000,
            "starttime": 17000,
            "nonce": None,
            "pgid": 731,
            "sid": 731,
            "cmdline": ["coli", "ramdisk", "start"],
            "state_dir": None,
            "weights_dir": None,
            "environment_candidates": {},
            "environment_ambiguities": (
                "COLI_MANAGED_NONCE",
                "COLI_STATE_DIR",
                "COLI_WEIGHTS_DIR",
            ),
        }
        with mock.patch.object(
            linux_ops, "_require_linux"
        ), mock.patch.object(
            linux_ops,
            "_proc_pid_snapshot",
            return_value={731: "/proc/731"},
        ), mock.patch.object(
            linux_ops,
            "_inspect_managed_launch_pid",
            return_value=observation,
        ):
            self.assertEqual(
                linux_ops._managed_launch_processes(
                    "a" * 48,
                    1000,
                    state_dir="/state/node-0",
                    weights_dir="/mnt/weights",
                    not_before_starttime=17000,
                    launcher_pid=731,
                    launcher_starttime=17000,
                    launcher_cmdline=["coli", "ramdisk", "start"],
                    expected_command=["coli", "serve"],
                ),
                [],
            )

    def test_managed_launch_scan_excludes_the_unattributed_recovery_process(self):
        recovery_pid = linux_ops.os.getpid()
        observation = {
            "kind": "same-uid",
            "pid": recovery_pid,
            "uid": 1000,
            "starttime": 18000,
            "nonce": None,
            "pgid": recovery_pid,
            "sid": recovery_pid,
            "cmdline": ["coli", "ramdisk", "start"],
            "state_dir": None,
            "weights_dir": None,
            "environment_candidates": {},
            "environment_ambiguities": (
                "COLI_MANAGED_NONCE",
                "COLI_STATE_DIR",
                "COLI_WEIGHTS_DIR",
            ),
        }
        snapshot = {recovery_pid: "/proc/%d" % recovery_pid}
        with mock.patch.object(
            linux_ops, "_require_linux"
        ), mock.patch.object(
            linux_ops, "_proc_pid_snapshot", return_value=snapshot
        ), mock.patch.object(
            linux_ops,
            "_inspect_managed_launch_pid",
            return_value=observation,
        ):
            self.assertEqual(
                linux_ops._managed_launch_processes(
                    "a" * 48,
                    1000,
                    state_dir="/state/node-0",
                    weights_dir="/mnt/weights",
                    not_before_starttime=17000,
                    launcher_pid=700,
                    launcher_starttime=16000,
                    launcher_cmdline=["coli", "ramdisk", "start"],
                    expected_command=["coli", "serve"],
                ),
                [],
            )

    def test_managed_launch_scan_rejects_turnover_after_final_reads(self):
        stable = {
            "kind": "same-uid",
            "pid": 731,
            "uid": 1000,
            "starttime": 17000,
            "nonce": None,
            "pgid": 731,
            "sid": 731,
            "cmdline": ["unrelated-worker"],
            "state_dir": None,
            "weights_dir": None,
            "environment_candidates": {},
            "environment_ambiguities": (),
        }
        with mock.patch.object(
            linux_ops, "_require_linux"
        ), mock.patch.object(
            linux_ops,
            "_proc_pid_snapshot",
            side_effect=[
                {731: "/proc/731"},
                {731: "/proc/731"},
                {731: "/proc/731"},
                {732: "/proc/732"},
            ],
        ), mock.patch.object(
            linux_ops,
            "_inspect_managed_launch_pid",
            side_effect=[stable, stable, None],
        ):
            with self.assertRaisesRegex(
                ramdisk.RamdiskError,
                "unverified PID.*732",
            ):
                linux_ops._managed_launch_processes(
                    "a" * 48,
                    1000,
                    state_dir="/state/node-0",
                    weights_dir="/mnt/weights",
                    not_before_starttime=17000,
                    launcher_pid=700,
                    launcher_starttime=16000,
                    launcher_cmdline=["coli", "ramdisk", "start"],
                    expected_command=["coli", "serve"],
                )

    def test_managed_launch_scan_rechecks_final_snapshot_identity(self):
        stable = {
            "kind": "same-uid",
            "pid": 731,
            "uid": 1000,
            "starttime": 17000,
            "nonce": None,
            "pgid": 731,
            "sid": 731,
            "cmdline": ["unrelated-worker"],
            "state_dir": None,
            "weights_dir": None,
        }
        reused = dict(stable, starttime=17001)
        with mock.patch.object(
            linux_ops, "_require_linux"
        ), mock.patch.object(
            linux_ops,
            "_proc_pid_snapshot",
            side_effect=[{731: "/proc/731"}] * 3,
        ), mock.patch.object(
            linux_ops,
            "_inspect_managed_launch_pid",
            side_effect=[stable, stable, reused],
        ):
            with self.assertRaisesRegex(
                ramdisk.RamdiskError,
                "changed during final",
            ):
                linux_ops._managed_launch_processes(
                    "a" * 48,
                    1000,
                    state_dir="/state/node-0",
                    weights_dir="/mnt/weights",
                    not_before_starttime=17000,
                    launcher_pid=700,
                    launcher_starttime=16000,
                    launcher_cmdline=["coli", "ramdisk", "start"],
                    expected_command=["coli", "serve"],
                )

    def test_managed_launch_scan_rechecks_same_pid_after_fourth_snapshot(self):
        stable = {
            "kind": "same-uid",
            "pid": 731,
            "uid": 1000,
            "starttime": 17000,
            "nonce": None,
            "pgid": 731,
            "sid": 731,
            "cmdline": ["unrelated-worker"],
            "state_dir": None,
            "weights_dir": None,
            "environment_candidates": {},
            "environment_ambiguities": (),
        }
        reused = dict(stable, starttime=17001)
        with mock.patch.object(
            linux_ops, "_require_linux"
        ), mock.patch.object(
            linux_ops,
            "_proc_pid_snapshot",
            return_value={731: "/proc/731"},
        ), mock.patch.object(
            linux_ops,
            "_inspect_managed_launch_pid",
            side_effect=[stable, stable, stable, reused],
        ):
            with self.assertRaisesRegex(
                ramdisk.RamdiskError,
                "changed after the final",
            ):
                linux_ops._managed_launch_processes(
                    "a" * 48,
                    1000,
                    state_dir="/state/node-0",
                    weights_dir="/mnt/weights",
                    not_before_starttime=17000,
                    launcher_pid=700,
                    launcher_starttime=16000,
                    launcher_cmdline=["coli", "ramdisk", "start"],
                    expected_command=["coli", "serve"],
                )

    def test_managed_launch_scan_rechecks_same_pid_in_final_confirmation(self):
        stable = {
            "kind": "same-uid",
            "pid": 731,
            "uid": 1000,
            "state": "S",
            "inert": False,
            "starttime": 17000,
            "nonce": None,
            "pgid": 731,
            "sid": 731,
            "cmdline": ["unrelated-worker"],
            "state_dir": None,
            "weights_dir": None,
            "environment_candidates": {},
            "environment_ambiguities": (),
        }
        reused = dict(stable, starttime=17001)
        with mock.patch.object(
            linux_ops, "_require_linux"
        ), mock.patch.object(
            linux_ops,
            "_proc_pid_snapshot",
            return_value={731: "/proc/731"},
        ), mock.patch.object(
            linux_ops,
            "_inspect_managed_launch_pid",
            side_effect=[stable, stable, stable, stable, reused],
        ):
            with self.assertRaisesRegex(
                ramdisk.RamdiskError,
                "changed during final pending-launch identity confirmation",
            ):
                linux_ops._managed_launch_processes(
                    "a" * 48,
                    1000,
                    state_dir="/state/node-0",
                    weights_dir="/mnt/weights",
                    not_before_starttime=17000,
                    launcher_pid=700,
                    launcher_starttime=16000,
                    launcher_cmdline=["coli", "ramdisk", "start"],
                    expected_command=["coli", "serve"],
                )

    def test_managed_launch_scan_treats_pid_disappearance_as_benign(self):
        owner_reads = 0

        def file_stat(path):
            nonlocal owner_reads
            self.assertEqual(path, "/proc/731")
            owner_reads += 1
            if owner_reads == 1:
                return mock.Mock(st_uid=1000)
            raise FileNotFoundError(errno.ENOENT, "process exited", path)

        with mock.patch.object(
            linux_ops, "_require_linux"
        ), mock.patch.object(
            linux_ops.os,
            "listdir",
            side_effect=(["731"], [], [], [], []),
        ), mock.patch.object(
            linux_ops.os, "stat", side_effect=file_stat
        ), mock.patch(
            "builtins.open",
            side_effect=FileNotFoundError(
                errno.ENOENT,
                "process exited",
                "/proc/731/stat",
            ),
        ):
            self.assertEqual(
                linux_ops._managed_launch_processes(
                    "a" * 48,
                    1000,
                    state_dir="/state/node-0",
                    weights_dir="/mnt/weights",
                    not_before_starttime=17000,
                    launcher_pid=700,
                    launcher_starttime=16000,
                    launcher_cmdline=["coli", "ramdisk", "start"],
                            expected_command=["coli", "serve"],
                ),
                [],
            )

    def test_managed_launch_scan_rejects_unreadable_or_malformed_same_uid(self):
        cases = (
            (
                "denied-environ",
                PermissionError(
                    errno.EACCES,
                    "permission denied",
                    "/proc/731/environ",
                ),
                "cannot read Linux process identity /proc/731/environ",
            ),
            (
                "malformed-environ",
                None,
                "ambiguous managed launch attribution",
            ),
            (
                "malformed-stat",
                None,
                "cannot parse Linux process identity /proc/731/stat",
            ),
        )
        for case, environ_error, message in cases:
            def open_proc(path, mode, *args, **kwargs):
                del mode, args, kwargs
                if path.endswith("/stat"):
                    if case == "malformed-stat":
                        return io.StringIO("731 (truncated) S 1\n")
                    return io.StringIO(
                        _proc_stat_record(731, 731, 731, 17001)
                    )
                if path.endswith("/cmdline"):
                    return io.BytesIO(b"coli\0serve\0")
                if path.endswith("/environ"):
                    if environ_error is not None:
                        raise environ_error
                    payload = (
                        b"not-an-environment-entry\0"
                        if case == "malformed-environ"
                        else b"OTHER=value\0"
                    )
                    return io.BytesIO(payload)
                raise AssertionError("unexpected open %s" % path)

            with self.subTest(case=case), mock.patch.object(
                linux_ops, "_require_linux"
            ), mock.patch.object(
                linux_ops.os, "listdir", return_value=["731"]
            ), mock.patch.object(
                linux_ops.os,
                "stat",
                return_value=mock.Mock(st_uid=1000),
            ), mock.patch(
                "builtins.open", side_effect=open_proc
            ):
                with self.assertRaisesRegex(ramdisk.RamdiskError, message):
                    linux_ops._managed_launch_processes(
                        "a" * 48,
                        1000,
                        state_dir="/state/node-0",
                        weights_dir="/mnt/weights",
                        not_before_starttime=17000,
                        launcher_pid=700,
                        launcher_starttime=16000,
                        launcher_cmdline=["coli", "ramdisk", "start"],
                            expected_command=["coli", "serve"],
                    )

    def test_managed_launch_scan_rejects_new_uninspected_pid(self):
        def open_proc(path, mode, *args, **kwargs):
            del mode, args, kwargs
            if path.endswith("/stat"):
                return io.StringIO(
                    _proc_stat_record(731, 731, 731, 17001)
                )
            if path.endswith("/cmdline"):
                return io.BytesIO(b"unrelated\0")
            if path.endswith("/environ"):
                return io.BytesIO(b"OTHER=value\0")
            raise AssertionError("unexpected open %s" % path)

        with mock.patch.object(
            linux_ops, "_require_linux"
        ), mock.patch.object(
            linux_ops.os,
            "listdir",
            side_effect=(["731"], ["731", "732"]),
        ), mock.patch.object(
            linux_ops.os,
            "stat",
            return_value=mock.Mock(st_uid=1000),
        ), mock.patch(
            "builtins.open", side_effect=open_proc
        ):
            with self.assertRaisesRegex(
                ramdisk.RamdiskError,
                "uninspected PID.*732",
            ):
                linux_ops._managed_launch_processes(
                    "a" * 48,
                    1000,
                    state_dir="/state/node-0",
                    weights_dir="/mnt/weights",
                    not_before_starttime=17000,
                    launcher_pid=700,
                    launcher_starttime=16000,
                    launcher_cmdline=["coli", "ramdisk", "start"],
                            expected_command=["coli", "serve"],
                )

    def test_managed_launch_scan_rejects_attribution_or_session_ambiguity(self):
        nonce = "a" * 48
        cases = (
            (
                "/wrong/state",
                731,
                "mismatched state or weights attribution",
            ),
            (
                "/state/node-0",
                700,
                "violates the new-session process-group identity",
            ),
        )
        for actual_state, session, message in cases:
            def open_proc(path, mode, *args, **kwargs):
                del mode, args, kwargs
                if path.endswith("/stat"):
                    return io.StringIO(
                        _proc_stat_record(731, 731, session, 17001)
                    )
                if path.endswith("/cmdline"):
                    return io.BytesIO(b"coli\0serve\0")
                if path.endswith("/environ"):
                    return io.BytesIO(
                        (
                            "COLI_MANAGED_NONCE=%s\0"
                            "COLI_STATE_DIR=%s\0"
                            "COLI_WEIGHTS_DIR=/mnt/weights\0"
                            % (nonce, actual_state)
                        ).encode()
                    )
                raise AssertionError("unexpected open %s" % path)

            with self.subTest(message=message), mock.patch.object(
                linux_ops, "_require_linux"
            ), mock.patch.object(
                linux_ops.os, "listdir", return_value=["731"]
            ), mock.patch.object(
                linux_ops.os,
                "stat",
                return_value=mock.Mock(st_uid=1000),
            ), mock.patch(
                "builtins.open", side_effect=open_proc
            ):
                with self.assertRaisesRegex(ramdisk.RamdiskError, message):
                    linux_ops._managed_launch_processes(
                        nonce,
                        1000,
                        state_dir="/state/node-0",
                        weights_dir="/mnt/weights",
                        not_before_starttime=17000,
                        launcher_pid=700,
                        launcher_starttime=16000,
                        launcher_cmdline=["coli", "ramdisk", "start"],
                            expected_command=["coli", "serve"],
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

    def test_process_group_liveness_treats_only_stable_zombies_as_inert(self):
        zombie = {
            "state": "Z",
            "starttime": 17000,
            "pgid": 731,
            "sid": 731,
            "num_threads": 1,
        }
        with mock.patch.object(
            linux_ops, "_require_linux"
        ), mock.patch.object(
            linux_ops.os, "killpg"
        ) as killpg, mock.patch.object(
            linux_ops,
            "_process_group_member_pids",
            side_effect=[[731], [731]],
        ), mock.patch.object(
            linux_ops.os,
            "stat",
            return_value=mock.Mock(st_uid=1000),
        ), mock.patch.object(
            linux_ops,
            "_strict_proc_stat_identity",
            return_value=zombie,
        ), mock.patch.object(
            linux_ops,
            "_stable_inert_process_identity",
            return_value=zombie,
        ):
            self.assertFalse(linux_ops._process_group_alive(731))

        killpg.assert_called_once_with(731, 0)

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

    def test_process_group_membership_churn_is_unverified_not_absent(self):
        zombie = {
            "pid": 731,
            "uid": 1000,
            "state": "Z",
            "inert": True,
            "starttime": 17000,
            "nonce": None,
            "pgid": 731,
            "sid": 731,
            "state_dir": None,
            "weights_dir": None,
        }
        child = {
            "pid": 732,
            "uid": 1000,
            "state": "S",
            "inert": False,
            "starttime": 17001,
            "nonce": "a" * 48,
            "pgid": 731,
            "sid": 731,
            "state_dir": "/state/node-0",
            "weights_dir": "/mnt/weights",
        }
        ops = mock.Mock()
        ops.process_group_member_pids.side_effect = (
            [731],
            [731],
            [731, 732],
            [731, 732],
        )
        identities = {731: zombie, 732: child}

        members, unreadable = processes._process_group_members(
            731,
            ops=ops,
            proc_identity=lambda pid: identities[pid],
        )

        self.assertTrue(unreadable)
        result = processes._process_matches(
            {
                "pid": 731,
                "pgid": 731,
                "uid": 1000,
                "starttime": 17000,
                "nonce": "a" * 48,
                "state_dir": "/state/node-0",
                "weights_dir": "/mnt/weights",
            },
            proc_identity=lambda ignored: zombie,
            process_group_members=lambda ignored: (members, unreadable),
        )
        self.assertFalse(result[0])
        self.assertEqual(result[1], "unverified-process-group")

    def test_process_group_stability_ignores_scheduler_state_transitions(self):
        member = {
            "pid": 731,
            "uid": 1000,
            "state": "R",
            "inert": False,
            "starttime": 17000,
            "nonce": "a" * 48,
            "pgid": 731,
            "sid": 731,
            "state_dir": "/state/node-0",
            "weights_dir": "/mnt/weights",
        }
        sleeping = dict(member, state="S")
        ops = mock.Mock()
        ops.process_group_member_pids.return_value = [731]

        members, unreadable = processes._process_group_members(
            731,
            ops=ops,
            proc_identity=mock.Mock(side_effect=(member, sleeping)),
        )

        self.assertEqual(members, [sleeping])
        self.assertEqual(unreadable, [])

    def test_pidfd_group_signal_opens_all_targets_before_validation(self):
        nonce = "a" * 48
        record = {
            "pid": 731,
            "pgid": 731,
            "uid": 1000,
            "starttime": 17000,
            "nonce": nonce,
            "state_dir": "/state/node-0",
            "weights_dir": "/mnt/weights",
        }
        identities = {
            731: dict(
                record,
                inert=False,
                state="S",
                sid=731,
            ),
            732: dict(
                record,
                pid=732,
                starttime=17001,
                inert=False,
                state="S",
                sid=731,
            ),
        }
        events = []

        def pidfd_open(pid, flags):
            events.append(("open", pid, flags))
            return pid + 1000

        def identity(pid):
            events.append(("identity", pid))
            return identities[pid]

        def pidfd_send(pidfd, signum, siginfo, flags):
            events.append(("signal", pidfd, signum, siginfo, flags))

        close = mock.Mock()
        with mock.patch.object(linux_ops, "_require_linux"), mock.patch.object(
            linux_ops.os, "killpg", create=True
        ) as killpg:
            result = linux_ops._signal_verified_process_group(
                record,
                signal.SIGTERM,
                process_group_member_pids=mock.Mock(
                    side_effect=([731, 732], [731, 732], [731, 732])
                ),
                process_identity=identity,
                process_group_alive=mock.Mock(return_value=True),
                pidfd_open=pidfd_open,
                pidfd_send_signal=pidfd_send,
                pidfd_exited=mock.Mock(return_value=False),
                close_fd=close,
            )

        self.assertEqual(result["status"], "signaled")
        self.assertEqual(result["signaled"], [731, 732])
        self.assertLess(
            max(index for index, event in enumerate(events) if event[0] == "open"),
            min(index for index, event in enumerate(events) if event[0] == "identity"),
        )
        self.assertLess(
            max(index for index, event in enumerate(events) if event[0] == "identity"),
            min(index for index, event in enumerate(events) if event[0] == "signal"),
        )
        killpg.assert_not_called()
        self.assertEqual(
            {call.args[0] for call in close.call_args_list},
            {1731, 1732},
        )

    def test_pidfd_group_signal_refuses_reused_leader_after_binding(self):
        record = {
            "pid": 731,
            "pgid": 731,
            "uid": 1000,
            "starttime": 17000,
            "nonce": "a" * 48,
            "state_dir": "/state/node-0",
            "weights_dir": "/mnt/weights",
        }
        reused = dict(
            record,
            starttime=17001,
            inert=False,
            state="S",
            sid=731,
        )
        send = mock.Mock()
        close = mock.Mock()
        with mock.patch.object(linux_ops, "_require_linux"):
            result = linux_ops._signal_verified_process_group(
                record,
                signal.SIGTERM,
                process_group_member_pids=mock.Mock(
                    side_effect=([731], [731], [731])
                ),
                process_identity=mock.Mock(return_value=reused),
                process_group_alive=mock.Mock(return_value=True),
                pidfd_open=mock.Mock(return_value=17),
                pidfd_send_signal=send,
                pidfd_exited=mock.Mock(return_value=False),
                close_fd=close,
            )

        self.assertEqual(result["status"], "foreign")
        self.assertEqual(result["reason"], "reused-pid")
        send.assert_not_called()
        close.assert_called_once_with(17)

    def test_pidfd_group_signal_treats_open_esrch_as_inconclusive(self):
        record = {
            "pid": 731,
            "pgid": 731,
            "uid": 1000,
            "starttime": 17000,
            "nonce": "a" * 48,
            "state_dir": "/state/node-0",
            "weights_dir": "/mnt/weights",
        }
        send = mock.Mock()
        with mock.patch.object(linux_ops, "_require_linux"):
            result = linux_ops._signal_verified_process_group(
                record,
                signal.SIGTERM,
                process_group_member_pids=mock.Mock(return_value=[731]),
                process_identity=mock.Mock(),
                process_group_alive=mock.Mock(return_value=True),
                pidfd_open=mock.Mock(side_effect=ProcessLookupError()),
                pidfd_send_signal=send,
                pidfd_exited=mock.Mock(return_value=False),
                close_fd=mock.Mock(),
            )

        self.assertEqual(result["status"], "inconclusive")
        send.assert_not_called()

    def test_pidfd_group_signal_detects_same_number_reuse_by_fd_readiness(self):
        record = {
            "pid": 731,
            "pgid": 731,
            "uid": 1000,
            "starttime": 17000,
            "nonce": "a" * 48,
            "state_dir": "/state/node-0",
            "weights_dir": "/mnt/weights",
        }
        exact = dict(record, inert=False, state="S", sid=731)
        send = mock.Mock()
        with mock.patch.object(linux_ops, "_require_linux"):
            result = linux_ops._signal_verified_process_group(
                record,
                signal.SIGTERM,
                process_group_member_pids=mock.Mock(return_value=[731]),
                process_identity=mock.Mock(return_value=exact),
                process_group_alive=mock.Mock(return_value=True),
                pidfd_open=mock.Mock(return_value=17),
                pidfd_send_signal=send,
                pidfd_exited=mock.Mock(
                    side_effect=(False, False, True)
                ),
                close_fd=mock.Mock(),
            )

        self.assertEqual(result["status"], "inconclusive")
        self.assertEqual(
            result["reason"],
            "pinned-member-exited-before-signal",
        )
        send.assert_not_called()

    def test_pidfd_group_signal_validates_entire_batch_before_first_signal(self):
        nonce = "a" * 48
        record = {
            "pid": 731,
            "pgid": 731,
            "uid": 1000,
            "starttime": 17000,
            "nonce": nonce,
            "state_dir": "/state/node-0",
            "weights_dir": "/mnt/weights",
        }
        leader = dict(record, inert=False, state="S", sid=731)
        foreign = dict(
            leader,
            pid=732,
            starttime=17001,
            nonce="b" * 48,
        )
        send = mock.Mock()
        with mock.patch.object(linux_ops, "_require_linux"):
            result = linux_ops._signal_verified_process_group(
                record,
                signal.SIGTERM,
                process_group_member_pids=mock.Mock(return_value=[731, 732]),
                process_identity=mock.Mock(side_effect=(leader, foreign)),
                process_group_alive=mock.Mock(return_value=True),
                pidfd_open=mock.Mock(side_effect=(17, 18)),
                pidfd_send_signal=send,
                pidfd_exited=mock.Mock(return_value=False),
                close_fd=mock.Mock(),
            )

        self.assertEqual(result["status"], "foreign")
        self.assertEqual(result["reason"], "foreign-nonce")
        send.assert_not_called()

    def test_pidfd_group_signal_reports_partial_non_esrch_failure(self):
        nonce = "a" * 48
        record = {
            "pid": 731,
            "pgid": 731,
            "uid": 1000,
            "starttime": 17000,
            "nonce": nonce,
            "state_dir": "/state/node-0",
            "weights_dir": "/mnt/weights",
        }
        identities = (
            dict(record, inert=False, state="S", sid=731),
            dict(
                record,
                pid=732,
                starttime=17001,
                inert=False,
                state="S",
                sid=731,
            ),
        )
        send = mock.Mock(
            side_effect=(None, OSError(errno.EIO, "input/output error"))
        )
        with mock.patch.object(linux_ops, "_require_linux"), mock.patch.object(
            linux_ops.os, "killpg", create=True
        ) as killpg:
            with self.assertRaisesRegex(
                ramdisk.RamdiskError,
                "after signaling PID\\(s\\) 731.*numeric fallback is forbidden",
            ):
                linux_ops._signal_verified_process_group(
                    record,
                    signal.SIGTERM,
                    process_group_member_pids=mock.Mock(
                        return_value=[731, 732]
                    ),
                    process_identity=mock.Mock(side_effect=identities),
                    process_group_alive=mock.Mock(return_value=True),
                    pidfd_open=mock.Mock(side_effect=(17, 18)),
                    pidfd_send_signal=send,
                    pidfd_exited=mock.Mock(return_value=False),
                    close_fd=mock.Mock(),
                )

        self.assertEqual(send.call_count, 2)
        killpg.assert_not_called()

    def test_pidfd_group_signal_fails_closed_when_kernel_lacks_pidfd(self):
        record = {"pid": 731, "pgid": 731}
        send = mock.Mock()
        with mock.patch.object(linux_ops, "_require_linux"):
            with self.assertRaisesRegex(
                ramdisk.RamdiskError,
                "requires Linux pidfd_open.*pidfd_open failed",
            ):
                linux_ops._signal_verified_process_group(
                    record,
                    signal.SIGTERM,
                    process_group_member_pids=mock.Mock(return_value=[731]),
                    process_identity=mock.Mock(),
                    process_group_alive=mock.Mock(return_value=True),
                    pidfd_open=mock.Mock(
                        side_effect=OSError(
                            errno.ENOSYS,
                            "function not implemented",
                        )
                    ),
                    pidfd_send_signal=send,
                    pidfd_exited=mock.Mock(return_value=False),
                    close_fd=mock.Mock(),
                )

        send.assert_not_called()

    def test_pidfd_group_signal_treats_send_esrch_as_gone(self):
        record = {
            "pid": 731,
            "pgid": 731,
            "uid": 1000,
            "starttime": 17000,
            "nonce": "a" * 48,
            "state_dir": "/state/node-0",
            "weights_dir": "/mnt/weights",
        }
        exact = dict(record, inert=False, state="S", sid=731)
        with mock.patch.object(linux_ops, "_require_linux"):
            result = linux_ops._signal_verified_process_group(
                record,
                signal.SIGTERM,
                process_group_member_pids=mock.Mock(return_value=[731]),
                process_identity=mock.Mock(return_value=exact),
                process_group_alive=mock.Mock(return_value=True),
                pidfd_open=mock.Mock(return_value=17),
                pidfd_send_signal=mock.Mock(
                    side_effect=ProcessLookupError()
                ),
                pidfd_exited=mock.Mock(return_value=False),
                close_fd=mock.Mock(),
            )

        self.assertEqual(result["status"], "signaled")
        self.assertEqual(result["signaled"], [])
        self.assertEqual(result["exited"], [731])

    def test_pidfd_group_signal_skips_exact_inert_members(self):
        nonce = "a" * 48
        record = {
            "pid": 731,
            "pgid": 731,
            "uid": 1000,
            "starttime": 17000,
            "nonce": nonce,
            "state_dir": "/state/node-0",
            "weights_dir": "/mnt/weights",
        }
        inert = {
            "pid": 731,
            "uid": 1000,
            "state": "Z",
            "inert": True,
            "starttime": 17000,
            "pgid": 731,
            "sid": 731,
        }
        live = dict(
            record,
            pid=732,
            starttime=17001,
            state="S",
            inert=False,
            sid=731,
        )
        send = mock.Mock()
        with mock.patch.object(linux_ops, "_require_linux"):
            result = linux_ops._signal_verified_process_group(
                record,
                signal.SIGTERM,
                process_group_member_pids=mock.Mock(return_value=[731, 732]),
                process_identity=mock.Mock(side_effect=(inert, live)),
                process_group_alive=mock.Mock(return_value=True),
                pidfd_open=mock.Mock(side_effect=(17, 18)),
                pidfd_send_signal=send,
                pidfd_exited=lambda pidfd: pidfd == 17,
                close_fd=mock.Mock(),
            )

        self.assertEqual(result["status"], "signaled")
        self.assertEqual(result["signaled"], [732])
        send.assert_called_once_with(18, signal.SIGTERM, None, 0)

    def test_pidfd_group_signal_accepts_only_stably_inert_absence(self):
        record = {
            "pid": 731,
            "pgid": 731,
            "uid": 1000,
            "starttime": 17000,
            "nonce": "a" * 48,
            "state_dir": "/state/node-0",
            "weights_dir": "/mnt/weights",
        }
        inert = {
            "pid": 731,
            "uid": 1000,
            "state": "Z",
            "inert": True,
            "starttime": 17000,
            "pgid": 731,
            "sid": 731,
        }
        send = mock.Mock()
        with mock.patch.object(linux_ops, "_require_linux"):
            result = linux_ops._signal_verified_process_group(
                record,
                signal.SIGTERM,
                process_group_member_pids=mock.Mock(return_value=[731]),
                process_identity=mock.Mock(return_value=inert),
                process_group_alive=mock.Mock(side_effect=(False, False)),
                pidfd_open=mock.Mock(return_value=17),
                pidfd_send_signal=send,
                pidfd_exited=mock.Mock(return_value=True),
                close_fd=mock.Mock(),
            )

        self.assertEqual(result["status"], "absent")
        send.assert_not_called()

    def test_verified_termination_retries_inconclusive_pidfd_scan(self):
        record = {"pid": 731, "pgid": 731}
        ops = mock.Mock()
        ops.signal_verified_process_group.side_effect = (
            {
                "status": "inconclusive",
                "reason": "membership-changed",
                "members": [731],
            },
            {"status": "signaled", "signaled": [731], "members": [731]},
            {"status": "absent", "members": []},
        )

        with mock.patch.object(processes.time, "sleep") as sleep:
            failure = processes._terminate_verified_group(
                record,
                term_seconds=1.0,
                kill_seconds=1.0,
                managed_child_liveness=lambda ignored: False,
                ops=ops,
            )

        self.assertIsNone(failure)
        self.assertEqual(ops.signal_verified_process_group.call_count, 3)
        self.assertTrue(
            all(
                call.args[1] == signal.SIGTERM
                for call in ops.signal_verified_process_group.call_args_list
            )
        )
        sleep.assert_called()

    def test_verified_termination_retries_transient_strict_identity_error(self):
        record = {
            "pid": 731,
            "pgid": 731,
            "uid": 1000,
            "starttime": 17000,
            "nonce": "a" * 48,
            "state_dir": "/state/node-0",
            "weights_dir": "/mnt/weights",
        }
        exact = dict(record, inert=False, state="S", sid=731)
        sends = []

        class Ops:
            calls = 0

            def signal_verified_process_group(self, current, signum):
                self.calls += 1
                if self.calls == 3:
                    members = mock.Mock(return_value=[])
                    identity = mock.Mock()
                    alive = mock.Mock(return_value=False)
                else:
                    members = mock.Mock(return_value=[731])
                    if self.calls == 1:
                        identity = mock.Mock(
                            side_effect=ramdisk.RamdiskError(
                                "procfs changed during TERM transition"
                            )
                        )
                    else:
                        identity = mock.Mock(return_value=exact)
                    alive = mock.Mock(return_value=True)
                return linux_ops._signal_verified_process_group(
                    current,
                    signum,
                    process_group_member_pids=members,
                    process_identity=identity,
                    process_group_alive=alive,
                    pidfd_open=mock.Mock(return_value=17),
                    pidfd_send_signal=lambda *args: sends.append(args),
                    pidfd_exited=mock.Mock(return_value=False),
                    close_fd=mock.Mock(),
                )

        ops = Ops()
        with mock.patch.object(linux_ops, "_require_linux"), mock.patch.object(
            processes.time, "sleep"
        ):
            failure = processes._terminate_verified_group(
                record,
                term_seconds=1.0,
                kill_seconds=1.0,
                managed_child_liveness=lambda ignored: False,
                ops=ops,
            )

        self.assertIsNone(failure)
        self.assertEqual(ops.calls, 3)
        self.assertEqual(len(sends), 1)
        self.assertEqual(sends[0][1], signal.SIGTERM)

    def test_verified_termination_resignals_new_exact_members(self):
        record = {"pid": 731, "pgid": 731}
        ops = mock.Mock()
        ops.signal_verified_process_group.side_effect = (
            {"status": "signaled", "signaled": [731], "members": [731]},
            {
                "status": "signaled",
                "signaled": [731, 732],
                "members": [731, 732],
            },
            {"status": "absent", "members": []},
        )

        with mock.patch.object(processes.time, "sleep"):
            failure = processes._terminate_verified_group(
                record,
                term_seconds=1.0,
                kill_seconds=1.0,
                managed_child_liveness=lambda ignored: False,
                ops=ops,
            )

        self.assertIsNone(failure)
        self.assertEqual(ops.signal_verified_process_group.call_count, 3)

    def test_verified_termination_fails_foreign_attribution_immediately(self):
        record = {"pid": 731, "pgid": 731}
        ops = mock.Mock()
        ops.signal_verified_process_group.return_value = {
            "status": "foreign",
            "reason": "reused-pid",
            "members": [731],
        }

        failure = processes._terminate_verified_group(
            record,
            term_seconds=1.0,
            kill_seconds=1.0,
            managed_child_liveness=lambda ignored: False,
            ops=ops,
        )

        self.assertIn("reused-pid", failure)
        ops.signal_verified_process_group.assert_called_once_with(
            record,
            signal.SIGTERM,
        )

    def test_process_match_revalidates_every_live_group_member_attribution(self):
        nonce = "a" * 48
        record = {
            "pid": 731,
            "pgid": 731,
            "uid": 1000,
            "starttime": 17000,
            "nonce": nonce,
            "state_dir": "/state/node-0",
            "weights_dir": "/mnt/weights",
        }
        leader = {
            "pid": 731,
            "uid": 1000,
            "starttime": 17000,
            "nonce": nonce,
            "pgid": 731,
            "sid": 731,
            "state_dir": "/state/node-0",
            "weights_dir": "/mnt/weights",
        }
        mismatches = (
            dict(leader, pid=732, pgid=999),
            dict(leader, pid=732, sid=999),
            dict(leader, pid=732, state_dir="/wrong/state"),
            dict(leader, pid=732, weights_dir="/wrong/weights"),
        )
        for member in mismatches:
            with self.subTest(member=member):
                result = processes._process_matches(
                    record,
                    proc_identity=lambda ignored: leader,
                    process_group_members=lambda ignored: ([member], []),
                )
                self.assertFalse(result[0])
                self.assertEqual(result[1], "foreign-process-group")

    def test_dead_wrapper_group_requires_exact_member_paths_and_session(self):
        nonce = "a" * 48
        record = {
            "pid": 731,
            "pgid": 731,
            "uid": 1000,
            "starttime": 17000,
            "nonce": nonce,
            "state_dir": "/state/node-0",
            "weights_dir": "/mnt/weights",
        }
        mismatched = {
            "pid": 732,
            "uid": 1000,
            "starttime": 17001,
            "nonce": nonce,
            "pgid": 999,
            "sid": 999,
            "state_dir": "/wrong/state",
            "weights_dir": "/wrong/weights",
        }
        result = processes._process_matches(
            record,
            proc_identity=lambda ignored: None,
            process_group_members=lambda ignored: ([mismatched], []),
        )
        self.assertFalse(result[0])
        self.assertEqual(result[1], "foreign-process-group")

    def test_inert_zombie_leader_does_not_hide_exact_live_child(self):
        nonce = "a" * 48
        record = {
            "pid": 731,
            "pgid": 731,
            "uid": 1000,
            "starttime": 17000,
            "nonce": nonce,
            "state_dir": "/state/node-0",
            "weights_dir": "/mnt/weights",
        }
        zombie = {
            "pid": 731,
            "uid": 1000,
            "state": "Z",
            "inert": True,
            "starttime": 17000,
            "nonce": None,
            "pgid": 731,
            "sid": 731,
            "state_dir": None,
            "weights_dir": None,
        }
        child = {
            "pid": 732,
            "uid": 1000,
            "state": "S",
            "inert": False,
            "starttime": 17001,
            "nonce": nonce,
            "pgid": 731,
            "sid": 731,
            "state_dir": "/state/node-0",
            "weights_dir": "/mnt/weights",
        }
        members = lambda ignored: ([zombie, child], [])
        running = processes._process_matches(
            record,
            proc_identity=lambda ignored: zombie,
            process_group_members=members,
        )
        self.assertTrue(running[0])
        self.assertEqual(running[1], "running-group")

        stopped = processes._process_matches(
            record,
            proc_identity=lambda ignored: zombie,
            process_group_members=lambda ignored: ([zombie], []),
        )
        self.assertEqual(stopped, (False, "not-running", None))

    def test_inert_nonleader_does_not_hide_exact_live_group_members(self):
        nonce = "a" * 48
        record = {
            "pid": 731,
            "pgid": 731,
            "uid": 1000,
            "starttime": 17000,
            "nonce": nonce,
            "state_dir": "/state/node-0",
            "weights_dir": "/mnt/weights",
        }
        leader = {
            "pid": 731,
            "uid": 1000,
            "inert": False,
            "starttime": 17000,
            "nonce": nonce,
            "pgid": 731,
            "sid": 731,
            "state_dir": "/state/node-0",
            "weights_dir": "/mnt/weights",
        }
        zombie = {
            "pid": 732,
            "uid": 1000,
            "state": "Z",
            "inert": True,
            "starttime": 17001,
            "nonce": None,
            "pgid": 731,
            "sid": 731,
            "state_dir": None,
            "weights_dir": None,
        }
        child = dict(leader, pid=733, starttime=17002)

        running = processes._process_matches(
            record,
            proc_identity=lambda ignored: leader,
            process_group_members=lambda ignored: ([leader, zombie, child], []),
        )

        self.assertTrue(running[0])
        self.assertEqual(running[1], "running")

    def test_all_inert_group_is_not_running_with_or_without_leader(self):
        record = {
            "pid": 731,
            "pgid": 731,
            "uid": 1000,
            "starttime": 17000,
            "nonce": "a" * 48,
            "state_dir": "/state/node-0",
            "weights_dir": "/mnt/weights",
        }
        leader = {
            "pid": 731,
            "uid": 1000,
            "state": "Z",
            "inert": True,
            "starttime": 17000,
            "nonce": None,
            "pgid": 731,
            "sid": 731,
            "state_dir": None,
            "weights_dir": None,
        }
        child = dict(leader, pid=732, starttime=17001)

        for case, actual, members in (
            ("leader-present", leader, [leader, child]),
            ("leader-absent", None, [child]),
        ):
            with self.subTest(case=case):
                stopped = processes._process_matches(
                    record,
                    proc_identity=lambda ignored, value=actual: value,
                    process_group_members=lambda ignored, value=members: (
                        value,
                        [],
                    ),
                )
                self.assertEqual(stopped, (False, "not-running", None))

        foreign = processes._process_matches(
            record,
            proc_identity=lambda ignored: None,
            process_group_members=lambda ignored: (
                [dict(child, uid=1001, pgid=999, sid=999)],
                [],
            ),
        )
        self.assertFalse(foreign[0])
        self.assertEqual(foreign[1], "foreign-process-group")

    def test_inert_group_member_requires_exact_uid_pgid_and_sid(self):
        nonce = "a" * 48
        record = {
            "pid": 731,
            "pgid": 731,
            "uid": 1000,
            "starttime": 17000,
            "nonce": nonce,
            "state_dir": "/state/node-0",
            "weights_dir": "/mnt/weights",
        }
        leader = {
            "pid": 731,
            "uid": 1000,
            "inert": False,
            "starttime": 17000,
            "nonce": nonce,
            "pgid": 731,
            "sid": 731,
            "state_dir": "/state/node-0",
            "weights_dir": "/mnt/weights",
        }
        zombie = {
            "pid": 732,
            "uid": 1000,
            "state": "Z",
            "inert": True,
            "starttime": 17001,
            "nonce": None,
            "pgid": 731,
            "sid": 731,
            "state_dir": None,
            "weights_dir": None,
        }
        for field, value in (("uid", 1001), ("pgid", 999), ("sid", 999)):
            with self.subTest(field=field):
                mismatched = dict(zombie, **{field: value})
                result = processes._process_matches(
                    record,
                    proc_identity=lambda ignored: leader,
                    process_group_members=lambda ignored: (
                        [leader, mismatched],
                        [],
                    ),
                )
                self.assertFalse(result[0])
                self.assertEqual(result[1], "foreign-process-group")

    def test_inert_leader_requires_exact_persisted_starttime(self):
        record = {
            "pid": 731,
            "pgid": 731,
            "uid": 1000,
            "starttime": 17000,
            "nonce": "a" * 48,
            "state_dir": "/state/node-0",
            "weights_dir": "/mnt/weights",
        }
        reused = {
            "pid": 731,
            "uid": 1000,
            "state": "Z",
            "inert": True,
            "starttime": 17001,
            "nonce": None,
            "pgid": 731,
            "sid": 731,
            "state_dir": None,
            "weights_dir": None,
        }
        child = {
            "pid": 732,
            "uid": 1000,
            "state": "S",
            "inert": False,
            "starttime": 17002,
            "nonce": "a" * 48,
            "pgid": 731,
            "sid": 731,
            "state_dir": "/state/node-0",
            "weights_dir": "/mnt/weights",
        }

        result = processes._process_matches(
            record,
            proc_identity=lambda ignored: reused,
            process_group_members=lambda ignored: ([reused, child], []),
        )

        self.assertFalse(result[0])
        self.assertEqual(result[1], "foreign-process-group")

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

        for reference in ("cwd", "root", "exe", "maps", "fd"):
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
                    "exe",
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
