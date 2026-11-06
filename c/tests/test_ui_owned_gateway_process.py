"""A UI-owned exited gateway must not consume the whole stop grace period.

Linux-only native /proc witness. No model, existing gateway or /proc mock is used.
"""
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

C = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(C))
spec = importlib.util.spec_from_file_location('owned_gateway_models', C.parent / 'ui/models.py')
models = importlib.util.module_from_spec(spec)
spec.loader.exec_module(models)


@unittest.skipUnless(sys.platform.startswith('linux'), 'native Linux /proc witness')
class OwnedGatewayProcessTests(unittest.TestCase):
    def test_exited_owned_gateway_does_not_wait_for_its_zombie(self):
        with tempfile.TemporaryDirectory(prefix='coli-owned-gateway-') as raw:
            root = Path(raw)
            log = root / 'gateway.log'
            process = models.launch(
                [sys.executable, '-u', '-c',
                 'import time; print("READY", flush=True); time.sleep(120)'],
                str(root), {}, str(log),
            )
            pid = process.pid
            # Match the successful switch worker: it leaves no Popen in a registry.
            del process
            completed = threading.Event()
            errors, signals = [], []
            real_kill = os.kill
            stopper = None

            def state():
                try:
                    return Path('/proc/%d/stat' % pid).read_text().rsplit(')', 1)[1].split()[0]
                except FileNotFoundError:
                    return None

            def signal_owned(target, sig):
                self.assertEqual(target, pid, 'refusing a signal to an unowned process')
                signals.append(int(sig))
                real_kill(target, sig)

            def stop():
                try:
                    models.stop_gateway({'pid': pid}, {})
                except BaseException as error:
                    errors.append(repr(error))
                finally:
                    completed.set()

            def reap_owned():
                # waitpid itself verifies parent ownership before any cleanup signal.
                try:
                    reaped, _ = os.waitpid(pid, os.WNOHANG)
                except ChildProcessError:
                    return
                if reaped:
                    return
                real_kill(pid, signal.SIGTERM)
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline:
                    reaped, _ = os.waitpid(pid, os.WNOHANG)
                    if reaped:
                        return
                    time.sleep(0.01)
                real_kill(pid, signal.SIGKILL)
                os.waitpid(pid, 0)

            patcher = patch.object(models.os, 'kill', side_effect=signal_owned)
            patcher.start()
            try:
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    if log.exists() and 'READY' in log.read_text():
                        break
                    time.sleep(0.01)
                else:
                    self.fail('owned stand-in did not become ready')
                stopper = threading.Thread(target=stop, daemon=True)
                began = time.monotonic()
                stopper.start()
                finished = completed.wait(3)
                witness = {'pid': pid, 'state': state(), 'stop_completed': finished,
                           'elapsed_seconds': time.monotonic() - began,
                           'errors': list(errors), 'signals': list(signals)}
                print('OWNED_GATEWAY_WITNESS ' + json.dumps(witness, sort_keys=True), flush=True)
                self.assertEqual(errors, [])
                self.assertEqual(signals, [int(signal.SIGTERM)])
                self.assertTrue(finished, 'stop_gateway still waits although its owned child is a native zombie')
                self.assertIsNone(state(), 'owned child must be reaped, not merely ignored')
            finally:
                try:
                    reap_owned()
                    if stopper is not None:
                        stopper.join(2)
                        self.assertFalse(stopper.is_alive(), 'owned stop worker outlived cleanup')
                    print('OWNED_GATEWAY_CLEANUP complete', flush=True)
                finally:
                    patcher.stop()

    def _state(self, pid):
        try:
            return Path('/proc/%d/stat' % pid).read_text().rsplit(')', 1)[1].split()[0]
        except FileNotFoundError:
            return None

    def _wait(self, condition, timeout=5):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if condition():
                return
            time.sleep(.01)
        self.fail('owned fixture condition timed out')

    def _cleanup(self, pid):
        try:
            reaped, _ = os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            return
        if not reaped:
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)

    def _child(self, root, mode):
        ready, release = root / ('ready-' + mode), root / ('release-' + mode)
        pid = os.fork()
        if not pid:
            try:
                if mode == 'ignore':
                    signal.signal(signal.SIGTERM, signal.SIG_IGN)
                elif mode == 'gate':
                    signal.signal(signal.SIGTERM, lambda *_: None)
                ready.touch()
                while not release.exists():
                    time.sleep(.01)
                os._exit(0)
            except BaseException:
                os._exit(99)
        self.addCleanup(self._cleanup, pid)
        self._wait(ready.exists)
        return pid, release

    def test_live_child_keeps_grace_until_exit(self):
        with tempfile.TemporaryDirectory() as raw:
            pid, release = self._child(Path(raw), 'gate')
            done = threading.Event()
            errors = []
            def stop():
                try:
                    models.stop_gateway({'pid': pid}, {})
                except BaseException as error:
                    errors.append(repr(error))
                finally:
                    done.set()
            worker = threading.Thread(target=stop, daemon=True)
            worker.start()
            try:
                self.assertFalse(done.wait(.25))
                self.assertNotEqual(self._state(pid), 'Z')
            finally:
                release.touch()
                worker.join(4)
                if worker.is_alive():
                    self._cleanup(pid)
                    worker.join(2)
            self.assertFalse(worker.is_alive())
            self.assertEqual(errors, [])
            self.assertIsNone(self._state(pid))

    def test_unrelated_child_status_remains_waitable(self):
        unrelated = os.fork()
        if not unrelated:
            os._exit(7)
        self.addCleanup(self._cleanup, unrelated)
        self._wait(lambda: self._state(unrelated) == 'Z')
        with tempfile.TemporaryDirectory() as raw:
            pid, _ = self._child(Path(raw), 'normal')
            models.stop_gateway({'pid': pid}, {})
        reaped, status = os.waitpid(unrelated, os.WNOHANG)
        self.assertEqual(reaped, unrelated)
        self.assertEqual(os.waitstatus_to_exitcode(status), 7)

    def test_reaping_keeps_descendant_fallback_and_cache_invalidation(self):
        # Discovery alone is controlled: fallback target is another DIRECT child,
        # so this fixture never leaves a grandchild for runner PID 1 to reap.
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            pid, _ = self._child(root, 'normal')
            child, _ = self._child(root, 'fallback')
            models._gw['at'] = 123
            with patch.object(models, 'proc_children', return_value=[child]):
                models.stop_gateway({'pid': pid}, {})
            self._wait(lambda: self._state(child) == 'Z')
            self.assertIsNone(self._state(pid))
            self.assertEqual(models._gw['at'], 0)
            self.assertEqual(os.waitpid(child, 0)[0], child)

    def test_ignored_sigterm_preserves_real_grace_and_escalation(self):
        with tempfile.TemporaryDirectory() as raw:
            pid, _ = self._child(Path(raw), 'ignore')
            began = time.monotonic()
            models.stop_gateway({'pid': pid}, {})
            elapsed = time.monotonic() - began
            self.assertGreaterEqual(elapsed, 59)
            reaped, status = os.waitpid(pid, 0)
            self.assertEqual(reaped, pid)
            self.assertTrue(os.WIFSIGNALED(status))
            self.assertEqual(os.WTERMSIG(status), signal.SIGKILL)
            print('REAL_GRACE_WITNESS ' + json.dumps({'seconds': elapsed, 'signal': signal.SIGKILL}), flush=True)

    def test_nonchild_gateway_uses_existing_proc_exit_path(self):
        import select
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            identity, ready = root / 'pid', root / 'ready'
            command_read, command_write = os.pipe()
            ack_read, ack_write = os.pipe()
            supervisor = os.fork()
            if not supervisor:
                os.close(command_write)
                os.close(ack_read)
                target = None
                stopping = [False]
                signal.signal(signal.SIGTERM, lambda *_: stopping.__setitem__(0, True))
                signal.signal(signal.SIGINT, lambda *_: stopping.__setitem__(0, True))
                exit_status = 98
                try:
                    cleaned = False
                    try:
                        if not stopping[0]:
                            target = os.fork()
                            if not target:
                                try:
                                    os.close(command_read)
                                    os.close(ack_write)
                                    signal.signal(signal.SIGTERM, signal.SIG_DFL)
                                    signal.signal(signal.SIGINT, signal.SIG_DFL)
                                    ready.touch()
                                    time.sleep(120)
                                finally:
                                    os._exit(0)
                            identity.write_text(str(target))
                        while target is not None and not stopping[0]:
                            reaped, _ = os.waitpid(target, os.WNOHANG)
                            if reaped:
                                target = None
                                break
                            readable, _, _ = select.select([command_read], [], [], .02)
                            if readable:
                                os.read(command_read, 1)  # command or parent EOF
                                break
                    finally:
                        if target is not None:
                            # This supervisor alone owns and reaps its recorded child.
                            reaped, _ = os.waitpid(target, os.WNOHANG)
                            if not reaped:
                                os.kill(target, signal.SIGKILL)
                                deadline = time.monotonic() + 3
                                while time.monotonic() < deadline:
                                    reaped, _ = os.waitpid(target, os.WNOHANG)
                                    if reaped:
                                        break
                                    time.sleep(.01)
                            cleaned = reaped == target
                        else:
                            cleaned = True
                        if cleaned:
                            os.write(ack_write, b'clean')
                        os.close(command_read)
                        os.close(ack_write)
                        exit_status = 0 if cleaned else 97
                finally:
                    # Forked supervisor must never unwind into unittest, including
                    # BrokenPipeError when its parent dies before the acknowledgment.
                    os._exit(exit_status)
            os.close(command_read)
            os.close(ack_write)
            closed = [False]
            def cleanup_supervisor():
                if closed[0]:
                    return
                closed[0] = True
                os.close(command_write)  # EOF requests target cleanup, never kills its parent.
                try:
                    readable, _, _ = select.select([ack_read], [], [], 5)
                    ack = os.read(ack_read, 5) if readable else b''
                    deadline = time.monotonic() + 2
                    reaped, status = 0, 0
                    while time.monotonic() < deadline:
                        reaped, status = os.waitpid(supervisor, os.WNOHANG)
                        if reaped:
                            break
                        time.sleep(.01)
                    self.assertEqual(ack, b'clean', 'supervisor did not confirm child reaping')
                    self.assertEqual(reaped, supervisor, 'supervisor did not exit within cleanup bound')
                    self.assertEqual(os.waitstatus_to_exitcode(status), 0)
                finally:
                    os.close(ack_read)
            self.addCleanup(cleanup_supervisor)
            try:
                self._wait(lambda: identity.exists() and ready.exists())
                pid = int(identity.read_text())
                with self.assertRaises(ChildProcessError):
                    os.waitpid(pid, os.WNOHANG)
                models.stop_gateway({'pid': pid}, {})
                self.assertIsNone(self._state(pid))
            finally:
                cleanup_supervisor()


if __name__ == '__main__':
    unittest.main()
