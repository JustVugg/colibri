"""Switch admission uses a current gateway snapshot and releases failed admissions.

The UI HTTP routes, jobs, health reads and stop/wait logic are real. Linux process
information and model discovery are controlled; each gateway is an owned stdlib
HTTP child, never a user's gateway or an actual model process.
"""
import errno
import http.client
from http.server import ThreadingHTTPServer
import importlib.util
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from contextlib import ExitStack

C = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(C))
spec = importlib.util.spec_from_file_location('ui_model_switch', C.parent / 'ui/models.py')
models = importlib.util.module_from_spec(spec)
spec.loader.exec_module(models)
import family_registry

CHILD = r'''
from http.server import BaseHTTPRequestHandler,HTTPServer
import json,sys
class Handler(BaseHTTPRequestHandler):
 def do_GET(self):
  data=json.dumps({'status':'ok','scheduler':{'active':0,'queued':0}}).encode()
  self.send_response(200);self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
 def log_message(self,*args):pass
server=HTTPServer(('127.0.0.1',int(sys.argv[1])),Handler)
print(server.server_port,flush=True)
server.serve_forever()
'''


@unittest.skipUnless(os.name == 'posix', 'owned POSIX process/pipe fixture for the Linux model switcher')
class ModelSwitchTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='coli-model-switch-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        for name in ('old', 'A', 'B', 'state', 'c'):
            (self.root / name).mkdir()
        (self.root / 'c/fixture-engine').touch()
        self.owned, self.workers, self.clients, self.launches, self.signals = {}, [], [], [], []
        self.current = None
        self.closing = self.gate_b = self.hold_ready = self.fail_next_start = False
        self.fail_model = None
        self.local = threading.local()
        self.paused, self.resume = threading.Event(), threading.Event()
        self.ready_entered, self.allow_ready = threading.Event(), threading.Event()
        self.real_lock = threading.Lock()
        self.server = self.serving = None
        self.real_kill, self.real_exists = os.kill, os.path.exists
        self.original_thread = threading.Thread
        self.original_wait_up = models.wait_up
        stack = ExitStack()
        self.addCleanup(stack.close)
        self.addCleanup(self.cleanup)
        fixture = self

        class Gate:
            def acquire(self, blocking=True):
                if fixture.gate_b and getattr(fixture.local, 'caller', None) == 'B':
                    fixture.paused.set()
                    if not fixture.resume.wait(12):
                        raise RuntimeError('fixture admission gate was not released')
                return fixture.real_lock.acquire(blocking=blocking)

            def release(self):
                fixture.real_lock.release()

        descriptor = SimpleNamespace(descriptor=SimpleNamespace(engine_artifact='fixture-engine', build_target='fixture-engine'))
        for target, name, options in (
            (models, 'gateway_proc', {'side_effect': lambda: self.current}),
            (models, 'COLI_DIR', {'new': str(self.root / 'c')}),
            (models, '_switching', {'new': Gate()}),
            (models, 'proc_children', {'return_value': []}),
            (models.os, 'kill', {'side_effect': self.safe_kill}),
            (models.os.path, 'exists', {'side_effect': self.proc_exists}),
            (models, 'launch', {'side_effect': self.launch}),
            (models, 'wait_up', {'side_effect': self.wait_up}),
            (models.threading, 'Thread', {'side_effect': self.thread}),
            (family_registry, 'resolve_model', {'return_value': descriptor}),
        ):
            stack.enter_context(patch.object(target, name, **options))
        models.JOBS.clear()
        initial, self.port = self.child(0)
        argv = [sys.executable, str(self.root / 'c/coli'), 'serve', '--model',
                str(self.root / 'old'), '--host', '127.0.0.1', '--port', str(self.port)]
        self.current = self.record(initial, argv)
        models.CFG.host, models.CFG.port = '127.0.0.1', self.port
        models.CFG.gateway = 'http://127.0.0.1:%d' % self.port
        models.CFG.dirs, models.CFG.state = [str(self.root)], str(self.root / 'state')

        class Handler(models.Handler):
            def _go(self):
                fixture.local.caller = self.headers.get('X-Fixture-Caller')
                try:
                    return super()._go()
                finally:
                    fixture.local.caller = None

            do_GET = do_POST = _go

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.server.daemon_threads = True
        self.serving = self.original_thread(target=self.server.serve_forever, args=(0.01,), daemon=True)
        self.serving.start()

    def cleanup(self):
        self.closing = True
        self.resume.set()
        self.allow_ready.set()
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
            self.serving.join(2)
        for client in self.clients:
            client.join(6)
        for worker in self.workers:
            worker.join(6)
        for process in self.owned.values():
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=3)
            process.stdout.close()
            process.stderr.close()
        for worker in self.workers:
            worker.join(3)
            self.assertFalse(worker.is_alive(), 'owned switch worker outlived fixture')
        self.assertTrue(all(not client.is_alive() for client in self.clients))

    def safe_kill(self, pid, sig):
        process = self.owned.get(pid)
        self.assertIsNotNone(process, 'refusing a signal to an unowned process')
        alive = process.poll() is None
        self.signals.append((pid, alive))
        if not alive:
            raise ProcessLookupError(errno.ESRCH, 'owned gateway already exited')
        self.real_kill(pid, sig)

    def proc_exists(self, path):
        text = os.fspath(path)
        if text.startswith('/proc/') and text[6:].isdigit():
            process = self.owned.get(int(text[6:]))
            self.assertIsNotNone(process, 'fixture never inspects unrelated processes')
            return process.poll() is None
        return self.real_exists(path)

    def child(self, port, fail=False):
        process = subprocess.Popen([sys.executable, '-u', '-c', 'raise SystemExit(7)' if fail else CHILD, str(port)],
                                   stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, env={}, start_new_session=True)
        self.owned[process.pid] = process
        if fail:
            return process, None
        self.assertTrue(select.select([process.stdout], [], [], 3)[0], 'fixture did not bind its port')
        line = process.stdout.readline()
        self.assertTrue(line, 'owned HTTP stand-in exited without a port')
        return process, int(line)

    def record(self, process, argv):
        return {'pid': process.pid, 'argv': list(argv), 'cwd': str(self.root),
                'env': {}, 'coli': True, 'key': 'fixture'}

    def launch(self, argv, cwd, env, log):
        if self.closing:
            raise RuntimeError('fixture is shutting down')
        if self.current:
            self.owned[self.current['pid']].wait(timeout=3)
        model = models._argval(argv, '--model')
        process, port = self.child(self.port, fail=Path(model).name == self.fail_model)
        self.current = self.record(process, argv)
        self.launches.append((process.pid, model))
        return process

    def wait_up(self, process, log, job, limit_s=1800):
        if self.hold_ready:
            self.ready_entered.set()
            if not self.allow_ready.wait(8):
                raise RuntimeError('fixture readiness gate was not released')
        return self.original_wait_up(process, log, job, limit_s)

    def thread(self, *args, **kwargs):
        thread = self.original_thread(*args, **kwargs)
        if getattr(kwargs.get('target'), '__name__', None) == 'go':
            if self.fail_next_start:
                self.fail_next_start = False

                def fail():
                    raise RuntimeError('fixture cannot start switch worker')

                thread.start = fail
            else:
                self.workers.append(thread)
        return thread

    def request(self, method, path, body=None, caller=''):
        connection = http.client.HTTPConnection(*self.server.server_address, timeout=10)
        try:
            connection.request(method, path, json.dumps(body) if body is not None else None,
                               {'X-Colibri-UI': '1', 'X-Fixture-Caller': caller})
            response = connection.getresponse()
            return response.status, json.loads(response.read())
        finally:
            connection.close()

    def use(self, name, caller=''):
        return self.request('POST', '/models/use', {'path': str(self.root / name)}, caller)

    def done(self, job_id):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            job = next(j for j in self.request('GET', '/models/jobs')[1]['jobs'] if j['id'] == job_id)
            if job['state'] in ('done', 'error'):
                return job
            time.sleep(0.01)
        self.fail('owned switch never finished')

    def successful_use(self, name):
        status, job = self.use(name)
        self.assertEqual(status, 200, job)
        result = self.done(job['id'])
        self.assertEqual(result['state'], 'done', result)
        return result

    def test_later_admission_uses_gateway_after_completed_switch(self):
        self.gate_b = True
        replies = []
        client = self.original_thread(target=lambda: replies.append(self.use('B', 'B')), daemon=True)
        self.clients.append(client)
        client.start()
        self.assertTrue(self.paused.wait(3))
        self.successful_use('A')
        self.resume.set()
        client.join(5)
        self.assertFalse(client.is_alive())
        self.assertEqual(replies[0][0], 200, replies)
        result = self.done(replies[0][1]['id'])
        self.assertEqual(result['state'], 'done', result)
        self.assertEqual(Path(self.launches[-1][1]).name, 'B')
        self.assertTrue(all(alive for _, alive in self.signals))

    def test_successful_sequential_switches(self):
        self.successful_use('A')
        self.successful_use('B')
        self.assertEqual([Path(model).name for _, model in self.launches], ['A', 'B'])

    def test_busy_switch_is_rejected_without_another_worker(self):
        self.hold_ready = True
        status, job = self.use('A')
        self.assertEqual(status, 200)
        self.assertTrue(self.ready_entered.wait(3))
        status, rejected = self.use('B')
        self.assertEqual(status, 409)
        self.assertIn('già cambiando', rejected['error'])
        self.assertEqual(len(self.launches), 1)
        self.allow_ready.set()
        self.assertEqual(self.done(job['id'])['state'], 'done')

    def test_rejected_preflight_releases_admission(self):
        self.current['coli'] = False
        status, rejected = self.use('A')
        self.assertEqual(status, 409)
        self.assertIn('altro processo', rejected['error'])
        self.current['coli'] = True
        self.successful_use('A')

    def test_worker_start_failure_releases_admission_and_marks_job(self):
        self.fail_next_start = True
        status, rejected = self.use('A')
        self.assertEqual(status, 409)
        self.assertIn('cannot start switch worker', rejected['error'])
        jobs = self.request('GET', '/models/jobs')[1]['jobs']
        self.assertEqual(self.launches, [])
        self.successful_use('B')
        self.assertEqual(jobs[0]['state'], 'error')

    def test_failed_new_model_restores_current_previous_model(self):
        self.successful_use('A')
        self.fail_model = 'B'
        status, job = self.use('B')
        self.assertEqual(status, 200)
        failed = self.done(job['id'])
        self.assertEqual(failed['state'], 'error')
        self.assertEqual(failed['note'], 'riavviato il modello di prima')
        self.assertEqual([Path(model).name for _, model in self.launches], ['A', 'B', 'A'])
        self.assertIsNone(self.owned[self.current['pid']].poll())


if __name__ == '__main__':
    unittest.main()
