"""Incomplete or mismatched campaigns must not become performance rankings."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import subprocess
import sys
from tests import test_benchmark_http_serving as http_fixture
import unittest
from unittest.mock import patch
from tools import benchmark_baseline as baseline
from tools import benchmark_http_serving as http


class BaselineTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        example = Path(__file__).resolve().parents[2] / 'docs/baselines/three-engine.example.json'
        self.spec = json.loads(example.read_text())
        self.spec.update(experiment='fixture', cache_policy='warm', reasoning_policy='off',
                         speculation_policy='off', quality_protocol='external fixture check')
        self.spec['hardware'] = dict.fromkeys(self.spec['hardware'], 'fixture host')
        sha = baseline.digest(example)
        self.spec['model'] = dict(source_revision='model-revision', tokenizer_sha256=sha,
                                  chat_template_sha256=sha)
        for engine in self.spec['engines'].values():
            engine.update(served_model='fixture', revision='engine-revision', launch_command='serve fixture',
                          artifact_sha256=sha, weight_format='fixture', quantization='fp32')
        self.spec['matrix'].update(concurrency=[1, 2], rounds=2, repeats=2, warmup_requests=1)
        self.manifest = self.root / 'manifest.json'
        (self.root / 'workload.jsonl').write_text('{"messages":[{"role":"user","content":"fixture"}]}\n')
        self.save_manifest()
        self.workload, self.sha = http.load_workload(self.root / 'workload.jsonl')
        self.results = self.root / 'results'

    def save_manifest(self):
        self.manifest.write_text(json.dumps(self.spec))

    @staticmethod
    def fake_run(**kwargs):
        rows = [dict(index=i, success=True, start_seconds=0, duration_seconds=2,
                     first_output_seconds=.5, completion_tokens=4, finish_reason='stop',
                     error=None, http_status=200)
                for i in range(len(kwargs['workload']) * kwargs['repeats'])]
        return rows, http.summarize(rows, 3, kwargs.get('slo_first_output'), kwargs.get('slo_duration'))

    def collect_all(self):
        with patch.object(http, 'run', side_effect=self.fake_run), contextlib.redirect_stdout(io.StringIO()):
            for item in baseline.plan(self.spec):
                baseline.collect(self.spec, self.workload, self.sha, item['engine'], item['round'], self.results)

    def change_report(self, change):
        path = self.results / 'colibri-r1-c1.json'
        data = json.loads(path.read_text())
        change(data)
        path.write_text(json.dumps(data))

    def compare(self):
        return baseline.compare(self.spec, self.workload, self.sha, self.results)

    def test_relative_workload_and_rotation(self):
        spec, workload, sha = baseline.load_manifest(self.manifest)
        self.assertEqual((sha, workload), (self.sha, self.workload))
        self.assertEqual([x['engine'] for x in baseline.plan(spec)],
                         ['colibri', 'sglang', 'vllm', 'sglang', 'vllm', 'colibri'])

    def test_artifact_mismatch_requires_deployment_mode(self):
        self.spec['engines']['vllm']['quantization'] = 'int4'
        self.spec['comparison'] = 'matched_artifact'
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, 'identical'):
            baseline.load_manifest(self.manifest)
        self.spec['comparison'] = 'deployment'
        self.save_manifest()
        baseline.load_manifest(self.manifest)

    def test_bad_matrix_and_placeholders(self):
        for key, value in [('concurrency', [True]), ('concurrency', [1, 1]),
                           ('concurrency', [16]), ('temperature', float('nan')), ('rounds', 0)]:
            spec = copy.deepcopy(self.spec)
            spec['matrix'][key] = value
            self.manifest.write_text(json.dumps(spec))
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                baseline.load_manifest(self.manifest)
        self.spec['hardware']['gpu'] = 'REPLACE-gpu'
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, 'gpu'):
            baseline.load_manifest(self.manifest)

    def test_complete_report_boundaries(self):
        self.collect_all()
        report = self.compare()
        self.assertEqual(report['status'], 'protocol_complete')
        self.assertEqual(report['quality'], 'not_assessed')
        self.assertEqual(report['token_latency'], 'not_measured')
        cell = report['cells'][0]
        self.assertEqual(cell['aggregate_completion_tokens_per_second']['median'], 8 / 3)
        self.assertEqual(cell['reported_output_tokens']['count'], 4)
        self.assertEqual(cell['slo_goodput_requests_per_second']['median'], 2 / 3)

    def test_missing_cell_is_not_dropped(self):
        self.collect_all()
        (self.results / 'colibri-r1-c1.json').unlink()
        report = self.compare()
        self.assertIn('missing colibri-r1-c1', report['issues'])
        self.assertIsNone(report['cells'][0]['aggregate_completion_tokens_per_second']['median'])
        self.assertEqual(report['cells'][0]['rounds_measured'], 1)

    def test_mismatched_and_modified_reports(self):
        mutations = [lambda d: d.update(workload_sha256='different'),
                     lambda d: d['manifest'].update(cache_policy='different'),
                     lambda d: d.update(harness_sha256='different'),
                     lambda d: d['summary'].update(successful_completion_tokens_per_second=999),
                     lambda d: d['requests'].pop()]
        self.collect_all()
        path = self.results / 'colibri-r1-c1.json'
        original = path.read_text()
        for mutate in mutations:
            path.write_text(original)
            self.change_report(mutate)
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                self.compare()

    def test_failed_empty_and_missing_usage(self):
        self.collect_all()
        def mutate(data):
            data['requests'][0].update(success=False, error='http_error')
            data['requests'][1].update(completion_tokens=None, first_output_seconds=None)
            data['summary'] = http.summarize(data['requests'], 3, 5, 120)
        self.change_report(mutate)
        report = self.compare()
        self.assertEqual(len(report['issues']), 3)
        self.assertIsNone(report['cells'][0]['aggregate_completion_tokens_per_second']['median'])

    def test_warmup_failure_and_no_overwrite(self):
        def fail(**kwargs):
            rows, _ = self.fake_run(**kwargs)
            rows[0]['success'] = False
            return rows, http.summarize(rows, 1)
        with patch.object(http, 'run', side_effect=fail) as run, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(baseline.collect(self.spec, self.workload, self.sha,
                                              'colibri', 1, self.results), 1)
            self.assertEqual(run.call_count, 1)
            with self.assertRaisesRegex(ValueError, 'already'):
                baseline.collect(self.spec, self.workload, self.sha, 'colibri', 1, self.results)
        data = json.loads((self.results / 'colibri-r1-c1.json').read_text())
        self.assertEqual(data['status'], 'warmup_failed')
        self.assertEqual(data['requests'], [])

    def test_secret_is_not_saved(self):
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'fixture-secret'}), \
                patch.object(http, 'run', side_effect=self.fake_run) as run, \
                contextlib.redirect_stdout(io.StringIO()):
            baseline.collect(self.spec, self.workload, self.sha, 'colibri', 1, self.results)
        self.assertEqual(run.call_args.kwargs['key'], 'fixture-secret')
        for path in self.results.glob('*.json'):
            self.assertNotIn('fixture-secret', path.read_text())

    def test_cli_collects_real_http_and_compare_reports_missing_cells(self):
        fixture = http_fixture.BenchmarkTest()
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        for config in self.spec['engines'].values():
            config['base_url'] = fixture.url.removesuffix('/chat/completions')
        self.spec['matrix'].update(rounds=1)
        self.save_manifest()
        command = [sys.executable, str(Path(baseline.__file__).resolve())]
        common = ['--manifest', str(self.manifest), '--results', str(self.results)]
        for engine in baseline.ENGINES:
            result = subprocess.run(command + ['run', '--engine', engine, '--round', '1'] + common,
                                    capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
        result = subprocess.run(command + ['compare'] + common, capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report['status'], 'protocol_complete')
        self.assertEqual(len(report['sources']), 6)
        self.assertEqual(len(fixture.server.payloads), 18)
        (self.results / 'vllm-r1-c2.json').unlink()
        result = subprocess.run(command + ['compare'] + common, capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn('missing vllm-r1-c2', json.loads(result.stdout)['issues'])
