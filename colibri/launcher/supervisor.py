"""Asynchronous ownership of one Colibri process tree, with local readiness."""
import codecs
from dataclasses import replace
import http.client
import json
import os
from pathlib import Path
import queue
import re
import signal
import socket
import subprocess
import threading
import time
from typing import Callable

from .domain import LaunchSpec, LauncherError, ProcessEvent
from .installation import external_environment, external_process_context

_MAX_LINE = 4096
_SECRETS = re.compile(
    r'''(?ix)(["']?(?:[\w-]*(?:api[_-]?key|password|passwd|secret|token)[\w-]*)["']?\s*[:=]\s*)'''
    r'''(?:"[^"\n]*"|'[^'\n]*'|[^\s,;}]+)'''
)
_BEARER = re.compile(r'(?i)(\bBearer\s+)\S+')
_AUTHORIZATION = re.compile(r'(?i)(\b(?:proxy-)?authorization\s*[:=]\s*)[^\r\n]+')


def _redact(text):
    text = _AUTHORIZATION.sub(r'\1[redacted]', text)
    return _BEARER.sub(r'\1[redacted]', _SECRETS.sub(r'\1[redacted]', text))


class Supervisor:
    """Callbacks run on a worker thread; GUI callers should queue their signals."""

    def __init__(self, callback: Callable[[ProcessEvent], None], poll_interval: float = 0.4):
        self._callback = callback
        self._interval = max(0.01, poll_interval)
        self._lock = threading.RLock()
        self._state = 'stopped'
        self._spec = None
        self._process = None
        self._thread = None
        self._cancel = threading.Event()
        self._done = threading.Event()
        self._done.set()

    @property
    def state(self):
        with self._lock:
            return self._state

    @property
    def pid(self):
        with self._lock:
            return self._process.pid if self._process and self._process.poll() is None else None

    @property
    def spec(self):
        with self._lock:
            return self._spec

    def start(self, spec: LaunchSpec) -> None:
        with self._lock:
            if not self._done.is_set():
                raise LauncherError('A model is already starting or running. Stop it first.')
            if not spec.argv or not 1 <= spec.port <= 65535:
                raise LauncherError('Choose a valid Python installation and a port from 1 to 65535.')
            self._spec = spec
            self._process = None
            self._cancel = threading.Event()
            self._done.clear()
            self._state = 'starting'
            self._thread = threading.Thread(target=self._run, args=(spec,), name='colibri-supervisor', daemon=True)
            self._thread.start()

    def stop(self) -> None:
        with self._lock:
            if not self._done.is_set():
                self._cancel.set()
                self._state = 'stopping'

    def wait(self, timeout: float = 10.0) -> bool:
        return self._done.wait(timeout)

    def _emit(self, kind, text):
        try:
            self._callback(ProcessEvent(kind, _redact(text)))
        except Exception:
            # A destroyed GUI must never strand the process it owned.
            self._cancel.set()

    def _state_event(self, state):
        with self._lock:
            self._state = state
        self._emit('state', state)

    @staticmethod
    def _check_port(port):
        with socket.socket() as sock:
            if os.name == 'nt':
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            try:
                sock.bind(('127.0.0.1', port))
            except OSError as exc:
                raise LauncherError(f'Port {port} is already in use or unavailable. Choose another port.') from exc

    def _ready(self, spec):
        # HTTPConnection never reads proxy variables or follows redirects.
        deadline = time.monotonic() + 1.0

        def read(path):
            connection = http.client.HTTPConnection('127.0.0.1', spec.port, timeout=0.3)
            finished = threading.Event()
            watcher = None
            response = None
            try:
                connection.connect()
                peer = connection.sock

                def interrupt_read():
                    # Socket timeouts only bound idle time. A peer can trickle
                    # bytes forever into HTTPResponse's header/body buffering.
                    # Shutdown also wakes those reads when makefile() still
                    # owns the socket after HTTPConnection has closed it.
                    while not finished.is_set():
                        if (self._cancel.is_set() or time.monotonic() >= deadline
                                or self._process.poll() is not None):
                            try:
                                peer.shutdown(socket.SHUT_RDWR)
                            except OSError:
                                pass
                            return
                        finished.wait(0.03)

                watcher = threading.Thread(target=interrupt_read, name='colibri-readiness', daemon=True)
                watcher.start()
                connection.request('GET', path)
                response = connection.getresponse()
                if response.status != 200:
                    return None
                body = response.read(65537)
                return json.loads(body) if len(body) <= 65536 else None
            finally:
                finished.set()
                if watcher is not None:
                    watcher.join()
                if response is not None:
                    response.close()
                connection.close()

        try:
            health = read('/health')
            if not isinstance(health, dict) or health.get('status') != 'ok':
                return False
            models = read('/v1/models')
            data = models.get('data') if isinstance(models, dict) else None
            return isinstance(data, list) and any(
                isinstance(model, dict) and model.get('id') == spec.model_id for model in data)
        except (OSError, ValueError, http.client.HTTPException):
            return False

    @staticmethod
    def _read_output(pipe, logs, evidence, port):
        decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
        pending = ''
        dropping = False
        marker = f'OpenAI-compatible API listening on http://127.0.0.1:{port}/v1'

        def submit(line, truncated=False):
            if line.strip() == marker:
                evidence.set()
            line = _redact(line)
            if truncated:
                line = line[:_MAX_LINE - 14] + ' … [truncated]'
            try:
                logs.put_nowait(line[:_MAX_LINE])
            except queue.Full:
                pass  # Draining pipes must never wait for a slow GUI.

        try:
            while True:
                chunk = os.read(pipe.fileno(), 4096)
                if not chunk:
                    pending += decoder.decode(b'', final=True)
                    if pending and not dropping:
                        submit(pending)
                    return
                text = decoder.decode(chunk)
                for part in text.splitlines(keepends=True):
                    end = part.endswith(('\n', '\r'))
                    if not dropping:
                        pending += part.rstrip('\r\n')
                        if len(pending) > _MAX_LINE:
                            submit(pending[:_MAX_LINE], truncated=True)
                            pending = ''
                            dropping = True
                    if end:
                        if not dropping:
                            submit(pending)
                        pending = ''
                        dropping = False
        except (OSError, ValueError):
            return

    @staticmethod
    def _group_alive(pgid):
        # Linux zombies are already dead, even though killpg(0) sees them.
        for entry in Path('/proc').iterdir():
            if not entry.name.isdigit():
                continue
            try:
                fields = (entry / 'stat').read_text().rsplit(') ', 1)[1].split()
                if int(fields[2]) == pgid and fields[0] != 'Z':
                    return True
            except (OSError, ValueError, IndexError):
                continue
        return False

    @classmethod
    def _stop_group(cls, process):
        def send(sig):
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                pass
        send(signal.SIGTERM)
        deadline = time.monotonic() + 3.0
        while cls._group_alive(process.pid) and time.monotonic() < deadline:
            process.poll()
            time.sleep(0.03)
        if cls._group_alive(process.pid):
            send(signal.SIGKILL)
            deadline = time.monotonic() + 2.0
            while cls._group_alive(process.pid) and time.monotonic() < deadline:
                process.poll()
                time.sleep(0.03)
            if cls._group_alive(process.pid):
                raise OSError('The operating system has not confirmed that the model processes exited')
        process.wait(timeout=1)

    def _run(self, spec):
        spec = replace(spec, env=external_environment(spec.env))
        process = None
        job = None
        reader = None
        logs = queue.Queue(maxsize=256)
        evidence = threading.Event()
        error = None
        try:
            self._emit('state', 'starting')
            if self._cancel.is_set():
                return
            self._check_port(spec.port)
            if self._cancel.is_set():
                return
            if os.name == 'nt':
                from .windows_job import WindowsJob
                job = WindowsJob()
                process = job.spawn(spec, self._cancel, self._lock)
            else:
                with external_process_context():
                    # Serialize the final cancellation check with process
                    # creation, so stop cannot return before a late spawn.
                    with self._lock:
                        if self._cancel.is_set():
                            return
                        process = subprocess.Popen(spec.argv, cwd=spec.cwd, env=spec.env,
                                                   stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                                   stderr=subprocess.STDOUT, start_new_session=True)
            with self._lock:
                self._process = process
            reader = threading.Thread(target=self._read_output,
                                      args=(process.stdout, logs, evidence, spec.port), daemon=True)
            reader.start()
            running = False
            while not self._cancel.is_set():
                self._drain(logs)
                code = process.poll()
                if code is not None:
                    raise LauncherError(f'Colibri exited with code {code}. Check its logs and installation, then try again.')
                if not running and evidence.is_set() and self._ready(spec):
                    with self._lock:
                        if not self._cancel.is_set() and process.poll() is None:
                            running = True
                            self._state_event('running')
                self._cancel.wait(self._interval)
        except (OSError, LauncherError, subprocess.SubprocessError) as exc:
            error = f'{exc} Check the selected installation and model settings, then try again.'
        finally:
            if process is not None:
                self._state_event('stopping')
                try:
                    if job is not None:
                        job.terminate()
                        process.wait(timeout=2)
                    else:
                        self._stop_group(process)
                except (OSError, subprocess.SubprocessError) as exc:
                    error = f'Could not confirm model shutdown: {exc}'
            if job is not None:
                job.close()
            if reader is not None:
                reader.join(timeout=1)
            if process is not None and process.stdout is not None:
                process.stdout.close()
            self._drain(logs)
            if error:
                self._emit('error', error)
            self._state_event('failed' if error else 'stopped')
            self._done.set()

    def _drain(self, logs):
        for _ in range(256):
            try:
                line = logs.get_nowait()
            except queue.Empty:
                break
            self._emit('log', line)
