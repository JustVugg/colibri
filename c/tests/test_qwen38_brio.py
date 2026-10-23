"""Pinned Qwen3.8 scores must survive unrelated requests and sibling branches.

Uses the generated tiny fixture and its 64-token byte tokenizer. All prompt
bytes are below 64; no unknown-token aliases can hide a divergent prefix.
QWEN38_TINY selects BF16 or FP8 fixtures. No full checkpoint is needed.
"""
import math
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
import unittest

HERE = Path(__file__).resolve().parent.parent
ENGINE = Path(os.environ.get('QWEN38_BINARY', HERE / ('qwen38.exe' if sys.platform == 'win32' else 'qwen38')))
FIXTURE = Path(os.environ.get('QWEN38_TINY', HERE / 'qwen38_tiny'))


class Engine:
    def __init__(self):
        env = dict(os.environ, SNAP=str(FIXTURE), SERVE='1',
                   TOK=str(FIXTURE / 'tokenizer.json'), OMP_NUM_THREADS='2',
                   COLI_NO_OMP_TUNE='1', COLI_CUDA='0', Q38_TRUNK_GPU='0',
                   Q38_MAXT='128')
        self.p = subprocess.Popen([str(ENGINE), '1', '8'], env=env,
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL)
        self.frames = queue.Queue()
        def read():
            try:
                while True:
                    raw = self.p.stdout.readline()
                    if not raw:
                        raise EOFError('engine closed')
                    fields = raw.decode().replace('\x01', '').split()
                    if fields and fields[0] in ('ECHO', 'DATA'):
                        n = int(fields[2])
                        if len(self.p.stdout.read(n)) != n or self.p.stdout.read(1) != b'\n':
                            raise EOFError('short protocol payload')
                    self.frames.put(fields)
            except Exception as exc:
                self.frames.put(exc)
        self.reader = threading.Thread(target=read, daemon=True)
        self.reader.start()
        try:
            deadline = time.monotonic() + 30
            while self.next(deadline)[0] != 'READY':
                pass
        except BaseException:
            self.close()
            raise

    def next(self, deadline):
        frame = self.frames.get(timeout=max(.01, deadline - time.monotonic()))
        if isinstance(frame, Exception):
            raise frame
        return frame

    def submit(self, rid, text, count=0, extension=' logprobs=1'):
        data = text.encode('ascii')
        assert all(b < 64 for b in data)
        self.p.stdin.write(f'SUBMIT {rid} 0 {len(data)} {count} 0 1{extension}\n'.encode()
                           + data + b'\n')
        self.p.stdin.flush()
        scores = {}
        accepted = False
        deadline = time.monotonic() + 30
        while True:
            f = self.next(deadline)
            if f[0] in ('ACCEPT', 'ECHO', 'DATA', 'DONE', 'ERROR'):
                assert f[1] == str(rid), f
            if f[0] == 'ACCEPT':
                assert int(f[2]) == len(data), f
                accepted = True
            elif f[0] == 'ECHO':
                score = float(f[4])
                assert math.isfinite(score), f
                scores[int(f[3])] = score
            elif f[0] == 'ERROR':
                raise AssertionError(f)
            elif f[0] == 'DONE':
                assert accepted
                return scores

    def close(self):
        self.p.terminate()
        try:
            self.p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.p.kill()
            self.p.wait(timeout=5)
        self.reader.join(timeout=5)
        self.p.stdin.close()
        self.p.stdout.close()


@unittest.skipUnless(ENGINE.is_file() and (FIXTURE / 'tokenizer.json').is_file(),
                     'build qwen38 and generate QWEN38_TINY with a byte tokenizer')
class Qwen38Brio(unittest.TestCase):
    def engine(self):
        e = Engine()
        self.addCleanup(e.close)
        return e

    def test_stale_pin_after_unrelated_generation(self):
        prefix, option = '123 456:', ' 0'
        for repetition in range(2):
            with self.subTest(repetition=repetition):
                e = self.engine()
                e.submit(1, prefix, extension=' logprobs=1 pin=1')
                warm = e.submit(2, prefix + option)
                e.submit(3, '789 012:', count=3, extension='')
                after = e.submit(4, prefix + option)
                fresh = self.engine().submit(1, prefix + option)
                tail = {8, 9}
                self.assertEqual(set(warm), tail, 'valid pins must still skip the prefix')
                self.assertTrue(tail <= set(after))
                self.assertEqual({p: warm[p] for p in tail}, {p: fresh[p] for p in tail})
                self.assertEqual({p: after[p] for p in tail}, {p: fresh[p] for p in tail})

    def test_shorter_branch_drops_old_tail(self):
        e = self.engine()
        e.submit(1, '123 456:', extension=' logprobs=1 pin=1')
        e.submit(2, '123 456: 01', extension=' logprobs=1 pin=1')
        e.submit(3, '123 456: ')
        after = e.submit(4, '123 456: 012')
        fresh = self.engine().submit(1, '123 456: 012')
        self.assertEqual(set(after), set(range(8, 12)))
        self.assertEqual(after, {p: fresh[p] for p in after})

    def test_nested_pins_and_sibling_branches(self):
        e = self.engine()
        e.submit(1, '123', extension=' logprobs=1 pin=1')
        e.submit(2, '123 456:', extension=' logprobs=1 pin=1')
        e.submit(3, '123 456: 0')
        # Same shallow pin, a different question overwrites the deeper pin's KV.
        sibling = e.submit(4, '123 789: 1')
        self.assertEqual(set(sibling), set(range(3, 10)))
        returned = e.submit(5, '123 456: 0')
        fresh = self.engine().submit(1, '123 456: 0')
        self.assertEqual(set(returned), set(range(3, 10)))
        self.assertEqual(returned, {p: fresh[p] for p in returned})


if __name__ == '__main__':
    unittest.main()
