"""Real subprocess tests for launcher ownership and readiness."""
import os
from pathlib import Path
import socket
import sys
import subprocess
import contextlib
import threading
import tempfile
import time
import unittest
from unittest.mock import patch
from dataclasses import replace

from colibri.launcher.domain import LaunchSpec, LauncherError
from colibri.launcher.supervisor import Supervisor

FIXTURE = Path(__file__).parent / 'fixtures' / 'fake_server.py'


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def eventually(predicate, timeout=6):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.02)
    return bool(predicate())


def alive(pid):
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        api = ctypes.WinDLL('kernel32', use_last_error=True)
        api.OpenProcess.restype = wintypes.HANDLE
        api.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        api.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        api.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = api.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        try:
            code = wintypes.DWORD()
            return bool(api.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
        finally:
            api.CloseHandle(handle)
    try:
        # A killed zombie cannot keep serving or spawn more descendants.
        return Path(f'/proc/{pid}/stat').read_text().split(') ')[1][0] != 'Z'
    except FileNotFoundError:
        return False


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.runner = Supervisor(self.events.append, poll_interval=0.03)
        self.addCleanup(self.cleanup)

    def cleanup(self):
        self.runner.stop()
        self.assertTrue(self.runner.wait(7), 'supervisor cleanup timed out')

    def spec(self, *extra, port=None):
        port = port or free_port()
        return LaunchSpec((sys.executable, '-u', str(FIXTURE.resolve()), '--port', str(port), *extra),
                          dict(os.environ), FIXTURE.parent.resolve(), port, 'test-model', 'serve', 'cpu')

    def test_delayed_start_is_nonblocking_then_stops_and_restarts(self):
        start = time.monotonic()
        self.runner.start(self.spec('--delay', '0.5'))
        self.assertLess(time.monotonic() - start, 0.3)
        self.assertEqual(self.runner.state, 'starting')
        self.assertTrue(eventually(lambda: self.runner.state == 'running'))
        self.runner.stop()
        self.assertTrue(self.runner.wait(6))
        self.assertEqual(self.runner.state, 'stopped')
        self.runner.start(self.spec())
        self.assertTrue(eventually(lambda: self.runner.state == 'running'))

    def test_double_start_rejected(self):
        self.runner.start(self.spec('--delay', '10'))
        with self.assertRaises(LauncherError):
            self.runner.start(self.spec())

    def test_stop_during_loading_cleans_child_and_grandchild(self):
        with tempfile.TemporaryDirectory() as directory:
            self.runner.start(self.spec('--delay', '30', '--tree', '2', '--pid-dir', directory))
            self.assertTrue(eventually(lambda: len(list(Path(directory).iterdir())) == 3))
            pids = [int(path.name) for path in Path(directory).iterdir()]
            self.assertTrue(all(alive(pid) for pid in pids))
            self.runner.stop()
            self.assertTrue(self.runner.wait(6))
            self.assertEqual(self.runner.state, 'stopped')
            self.assertTrue(eventually(lambda: not any(alive(pid) for pid in pids)))

    def test_immediate_cancellation_never_launches_late(self):
        with tempfile.TemporaryDirectory() as directory:
            self.runner.start(self.spec('--delay', '30', '--pid-dir', directory))
            self.runner.stop()
            self.assertTrue(self.runner.wait(6))
            pids = [int(path.name) for path in Path(directory).iterdir()]
            self.assertFalse(any(alive(pid) for pid in pids))
            self.assertEqual(self.runner.state, 'stopped')

    def test_occupied_port_preserves_unrelated_listener(self):
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            sock.listen()
            self.runner.start(self.spec(port=sock.getsockname()[1]))
            self.assertTrue(self.runner.wait(6))
            self.assertEqual(self.runner.state, 'failed')
            with socket.create_connection(sock.getsockname(), timeout=1):
                pass
            self.assertFalse(any(e.text == 'running' for e in self.events))

    def test_bad_responses_never_ready(self):
        for response in ('empty', 'wrong', 'bad-health', 'http-error', 'redirect', 'malformed'):
            with self.subTest(response=response):
                self.runner.start(self.spec('--response', response))
                time.sleep(0.45)
                self.assertEqual(self.runner.state, 'starting')
                self.runner.stop()
                self.assertTrue(self.runner.wait(6))
        self.assertFalse(any(e.text == 'running' for e in self.events))

    def test_endpoint_alone_without_owned_startup_evidence_never_ready(self):
        self.runner.start(self.spec('--no-evidence'))
        time.sleep(0.5)
        self.assertEqual(self.runner.state, 'starting')

    def test_exit_before_ready_reports_failure_and_cleans_descendants(self):
        with tempfile.TemporaryDirectory() as directory:
            self.runner.start(self.spec('--delay', '0.2', '--fail', '--tree', '2', '--pid-dir', directory))
            self.assertTrue(self.runner.wait(6))
            self.assertEqual(self.runner.state, 'failed')
            pids = [int(path.name) for path in Path(directory).iterdir()]
            self.assertFalse(any(alive(pid) for pid in pids))
            self.assertTrue(any(e.kind == 'error' and '7' in e.text for e in self.events))

    def test_exit_after_ready_reports_failure(self):
        self.runner.start(self.spec('--exit-after', '0.3'))
        self.assertTrue(eventually(lambda: self.runner.state == 'running'))
        self.assertTrue(self.runner.wait(6))
        self.assertEqual(self.runner.state, 'failed')

    def test_missing_executable_is_recoverable_failure(self):
        self.runner.start(replace(self.spec(), argv=('Z:/missing-colibri/python.exe',)))
        self.assertTrue(self.runner.wait(6))
        self.assertEqual(self.runner.state, 'failed')
        self.assertTrue(any(e.kind == 'error' for e in self.events))

    def test_large_output_is_bounded_decoded_and_redacted(self):
        self.runner.start(self.spec('--flood'))
        self.assertTrue(eventually(lambda: self.runner.state == 'running'))
        logs = [e.text for e in self.events if e.kind == 'log']
        self.assertTrue(logs)
        self.assertLessEqual(max(map(len, logs)), 4096)
        self.assertNotIn('very-secret-value', '\n'.join(logs))
        self.assertNotIn('another-secret', '\n'.join(logs))
        self.assertNotIn('x' * 100, '\n'.join(logs))

    def test_callback_failure_does_not_leak_process(self):
        with tempfile.TemporaryDirectory() as directory:
            def callback(event):
                if event.kind == 'log':
                    raise RuntimeError('GUI was destroyed')
            self.runner = Supervisor(callback, poll_interval=0.03)
            self.runner.start(self.spec('--pid-dir', directory))
            self.assertTrue(self.runner.wait(6))
            pids = [int(path.name) for path in Path(directory).iterdir()]
            self.assertFalse(any(alive(pid) for pid in pids))

    def check_stop_during_stream(self, response):
        with tempfile.TemporaryDirectory() as directory:
            pids_dir = Path(directory, 'pids')
            pids_dir.mkdir()
            marker = Path(directory, 'request')
            self.runner.start(self.spec('--response', response, '--request-marker', str(marker),
                                        '--tree', '2', '--pid-dir', str(pids_dir)))
            self.assertTrue(eventually(marker.exists))
            pids = [int(path.name) for path in pids_dir.iterdir()]
            self.runner.stop()
            try:
                self.assertTrue(self.runner.wait(4), 'Stop was blocked by a slow HTTP response')
                self.assertEqual(self.runner.state, 'stopped')
                self.assertFalse(any(alive(pid) for pid in pids))
            finally:
                # Also clean up on the expected RED run, once the old read ends.
                self.runner.wait(10)

    def test_stop_interrupts_slow_response_body_and_cleans_descendants(self):
        self.check_stop_during_stream('slow-body')

    def test_stop_interrupts_slow_response_header_and_cleans_descendants(self):
        self.check_stop_during_stream('slow-header')

    def test_slow_response_has_absolute_deadline_without_cancelling_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory, 'requests')
            self.runner.start(self.spec('--response', 'slow-body', '--request-marker', str(marker)))
            self.assertTrue(eventually(marker.exists))
            try:
                self.assertTrue(eventually(lambda: len(marker.read_text().splitlines()) >= 2, timeout=3),
                                'Readiness attempt never reached its elapsed-time deadline')
                self.assertEqual(self.runner.state, 'starting')
            finally:
                self.runner.stop()
                self.runner.wait(10)

    @unittest.skipUnless(os.name == 'nt', 'Windows bootstrap authorization gate')
    def test_cancel_after_job_assignment_before_gate_release_never_starts_engine(self):
        from colibri.launcher import windows_job
        entered, release = threading.Event(), threading.Event()
        encode = windows_job.json.dumps
        real_spawn = windows_job.WindowsJob.spawn
        with tempfile.TemporaryDirectory() as directory:
            def paused_encode(value):
                entered.set()
                release.wait(3)
                return encode(value)
            def scheduled_spawn(job, *args):
                process = real_spawn(job, *args)
                # Allow a wrongly released real child to record that it started
                # before the supervisor returns from spawn and cleans the Job.
                eventually(lambda: bool(list(Path(directory).iterdir())), timeout=0.5)
                return process
            with patch.object(windows_job.json, 'dumps', paused_encode):
                with patch.object(windows_job.WindowsJob, 'spawn', scheduled_spawn):
                    self.runner.start(self.spec('--delay', '30', '--pid-dir', directory))
                    self.assertTrue(entered.wait(3))
                    start = time.monotonic()
                    self.runner.stop()
                    self.assertLess(time.monotonic() - start, 0.3)
                    self.assertEqual(list(Path(directory).iterdir()), [])
                    release.set()
                    self.assertTrue(self.runner.wait(6))
            self.assertEqual(list(Path(directory).iterdir()), [], 'Engine started after Stop returned')

    def test_secret_fields_and_authorization_are_redacted_and_invalid_utf8_replaced(self):
        self.runner.start(self.spec('--secrets'))
        self.assertTrue(eventually(lambda: self.runner.state == 'running'))
        logs = '\n'.join(e.text for e in self.events if e.kind == 'log')
        self.assertIn('[redacted]', logs)
        self.assertNotIn('dXNlcjpwYXNzd29yZA==', logs)
        self.assertNotIn('hidden-token', logs)
        self.assertNotIn('hidden-key', logs)
        self.assertIn('invalid:\ufffd', logs)

    @unittest.skipUnless(os.name == 'nt', 'Windows Job Object API')
    def test_job_assignment_failure_never_starts_engine(self):
        from colibri.launcher.windows_job import WindowsJob
        original = WindowsJob.__init__
        def fail_assignment(job):
            original(job)
            job._api.AssignProcessToJobObject = lambda *args: 0
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(WindowsJob, '__init__', fail_assignment):
                self.runner.start(self.spec('--pid-dir', directory))
                self.assertTrue(self.runner.wait(6))
            self.assertEqual(self.runner.state, 'failed')
            self.assertEqual(list(Path(directory).iterdir()), [])

    @unittest.skipUnless(os.name == 'nt', 'Windows process creation context')
    def test_failed_dll_context_restore_cleans_waiting_bootstrap(self):
        from colibri.launcher import windows_job
        created = []
        real_popen = subprocess.Popen
        def track(*args, **kwargs):
            process = real_popen(*args, **kwargs)
            created.append(process)
            return process
        @contextlib.contextmanager
        def broken_context():
            yield
            raise OSError('cannot restore DLL search directory')
        with patch.object(windows_job, 'external_process_context', broken_context):
            with patch.object(windows_job.subprocess, 'Popen', track):
                self.runner.start(self.spec())
                self.assertTrue(self.runner.wait(6))
        try:
            self.assertEqual(self.runner.state, 'failed')
            self.assertEqual(len(created), 1)
            self.assertTrue(eventually(lambda: created[0].poll() is not None, timeout=1))
        finally:
            for process in created:
                if process.poll() is None:
                    process.kill()
                    process.wait()
                process.stdin.close()
                process.stdout.close()

    @unittest.skipIf(os.name == 'nt', 'Linux process creation path')
    def test_cancel_while_waiting_to_spawn_does_not_start_a_late_process(self):
        from colibri.launcher import supervisor
        entered, release = threading.Event(), threading.Event()
        @contextlib.contextmanager
        def slow_context():
            entered.set()
            release.wait(3)
            yield
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(supervisor, 'external_process_context', slow_context):
                self.runner.start(self.spec('--pid-dir', directory))
                self.assertTrue(entered.wait(2))
                self.runner.stop()
                release.set()
                self.assertTrue(self.runner.wait(6))
            self.assertEqual(list(Path(directory).iterdir()), [])

    @unittest.skipUnless(os.name == 'nt', 'Windows kill-on-close Job Object')
    def test_windows_job_closes_when_supervising_app_exits(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = self.spec('--delay', '30', '--tree', '2', '--pid-dir', directory)
            host_code = '''import os, sys, time
from pathlib import Path
from colibri.launcher.domain import LaunchSpec
from colibri.launcher.supervisor import Supervisor
argv = tuple(sys.argv[3:])
s = Supervisor(lambda e: None, poll_interval=0.03)
s.start(LaunchSpec(argv, dict(os.environ), Path.cwd(), int(sys.argv[1]), 'test-model', 'serve', 'cpu'))
end = time.monotonic() + 5
while len(list(Path(sys.argv[2]).iterdir())) < 3 and time.monotonic() < end:
    time.sleep(0.02)
os._exit(0)
'''
            host = subprocess.Popen([sys.executable, '-c', host_code, str(spec.port), directory, *spec.argv],
                                    creationflags=subprocess.CREATE_NO_WINDOW)
            self.addCleanup(lambda: host.kill() if host.poll() is None else None)
            host.wait(timeout=7)
            pids = [int(path.name) for path in Path(directory).iterdir()]
            self.assertEqual(len(pids), 3)
            self.assertTrue(eventually(lambda: not any(alive(pid) for pid in pids)))


if __name__ == '__main__':
    unittest.main()
