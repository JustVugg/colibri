"""Models job identities survive simultaneous submissions and cancel independently."""
import concurrent.futures
import http.client
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from http.server import ThreadingHTTPServer

C = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(C))
SPEC = importlib.util.spec_from_file_location('ui_model_jobs', C.parent / 'ui/models.py')
models = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(models)


class ModelJobIdentityTests(unittest.TestCase):
    def setUp(self):
        models.JOBS.clear()
        self.addCleanup(models.JOBS.clear)

    def test_parallel_jobs_keep_distinct_registry_entries(self):
        with patch.object(models, 'time', SimpleNamespace(time=lambda: 1700000000.123)):
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                jobs = list(pool.map(lambda i: models.new_job('download', str(i)), range(64)))
        self.assertEqual(len({j['id'] for j in jobs}), 64)
        self.assertEqual(len(models.JOBS), 64)
        for job in jobs:
            self.assertIs(models.JOBS[job['id']], job)
            self.assertRegex(job['id'], r'^[0-9a-f]+$')
            self.assertEqual(job['started'], 1700000000.123)

    def test_http_downloads_in_same_millisecond_list_and_cancel_independently(self):
        parked = threading.Event()
        condition = threading.Condition()
        workers, threads = [], []

        def download(job, *args):
            with condition:
                workers.append(job)
                threads.append(threading.current_thread())
                condition.notify_all()
            parked.wait(5)

        def entry(name):
            return SimpleNamespace(id=name, name=name, repo='fixture/' + name,
                revision='main', include=(), exclude=(), post_install=())

        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(models, 'time', SimpleNamespace(time=lambda: 1700000000.123)), \
                patch.object(models, 'roots', return_value=[tmp]), \
                patch.object(models.setup_catalog, 'by_id', side_effect=entry), \
                patch.object(models.setup_download, 'hf_token', return_value=None), \
                patch.object(models.setup_download, 'list_repo_files', return_value=[]), \
                patch.object(models, 'run_download', side_effect=download):
            server = ThreadingHTTPServer(('127.0.0.1', 0), models.Handler)
            server.daemon_threads = True
            serve = threading.Thread(target=server.serve_forever, daemon=True)
            serve.start()

            def request(method, path, body=None):
                conn = http.client.HTTPConnection(*server.server_address, timeout=3)
                try:
                    conn.request(method, path, json.dumps(body) if body is not None else None,
                                 {'X-Colibri-UI': '1'})
                    response = conn.getresponse()
                    return response.status, json.loads(response.read())
                finally:
                    conn.close()

            try:
                status, first = request('POST', '/models/download', {'id': 'first'})
                self.assertEqual(status, 200)
                status, second = request('POST', '/models/download', {'id': 'second'})
                self.assertEqual(status, 200)
                with condition:
                    self.assertTrue(condition.wait_for(lambda: len(workers) == 2, timeout=2))
                self.assertNotEqual(first['id'], second['id'])
                status, listed = request('GET', '/models/jobs')
                self.assertEqual(status, 200)
                self.assertEqual({j['id'] for j in listed['jobs']}, {first['id'], second['id']})
                self.assertTrue(all('cancel' not in j for j in listed['jobs']))
                by_title = {j['title']: j for j in workers}
                self.assertEqual(request('POST', '/models/jobs/' + first['id'] + '/cancel', {})[0], 200)
                self.assertTrue(by_title['first']['cancel'])
                self.assertFalse(by_title['second']['cancel'])
                self.assertEqual(request('POST', '/models/jobs/' + second['id'] + '/cancel', {})[0], 200)
                self.assertTrue(by_title['second']['cancel'])
                self.assertEqual(request('POST', '/models/jobs/deadbeef/cancel', {})[0], 404)
            finally:
                parked.set()
                server.shutdown()
                server.server_close()
                serve.join(timeout=2)
                for thread in threads:
                    thread.join(timeout=2)
                    self.assertFalse(thread.is_alive())
                self.assertFalse(serve.is_alive())

    def test_a_download_job_shows_each_retry_until_bytes_arrive(self):
        """#2970: the page said nothing while a download retried, until its final error."""
        job = models.new_job('download', 'fixture')
        seen = {}

        def pull(*args, progress, notice, **kwargs):
            notice('model.safetensors: Temporary failure in name resolution; trying again in 1 s '
                   '(retry 1 of 5)')
            seen['waiting'] = job['note']
            progress({'file': 'model.safetensors', 'done': 4, 'total': 10, 'fresh': 4})
            seen['moving'] = job['note']
            progress({'file': 'model.safetensors', 'done': 10, 'total': 10, 'fresh': 10})

        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(models.setup_download, 'hf_token', return_value=None), \
                patch.object(models.setup_download, 'bytes_present', return_value=0), \
                patch.object(models.setup_download, 'download_repo', side_effect=pull):
            models.run_download(job, 'fixture/m', 'main', [{'path': 'model.safetensors', 'size': 10}], tmp)
        self.assertEqual(job['state'], 'done', job['msg'])
        self.assertIn('Temporary failure in name resolution', seen['waiting'])
        self.assertEqual(seen['moving'], '')
        self.assertEqual(job['note'], '')

    def test_a_download_that_stops_while_retrying_drops_the_retry_line(self):
        def failing(*args, notice, **kwargs):
            notice('model.safetensors: Temporary failure; trying again in 15 s (retry 5 of 5)')
            raise models.setup_download.DownloadError('model.safetensors: Temporary failure '
                                                      '(rerun to continue from 0 of 10 bytes)')

        def cancelled(*args, progress, notice, **kwargs):
            notice('model.safetensors: no bytes from curl.exe for 60 s')
            job['cancel'] = True
            progress({'file': 'model.safetensors', 'done': 0, 'total': 10, 'fresh': 0})

        for pull, state in ((failing, 'error'), (cancelled, 'cancelled')):
            job = models.new_job('download', 'fixture')
            with tempfile.TemporaryDirectory() as tmp, \
                    patch.object(models.setup_download, 'hf_token', return_value=None), \
                    patch.object(models.setup_download, 'bytes_present', return_value=0), \
                    patch.object(models.setup_download, 'download_repo', side_effect=pull):
                models.run_download(job, 'fixture/m', 'main', [{'path': 'model.safetensors', 'size': 10}], tmp)
            self.assertEqual(job['state'], state)
            self.assertEqual(job['note'], '', state)


if __name__ == '__main__':
    unittest.main()
