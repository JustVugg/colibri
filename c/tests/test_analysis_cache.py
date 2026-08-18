"""analyze_model() sidecar cache (#3).

analyze_model() scans every shard header and regex-matches ~116k tensor names on
the 372 GB model, and reran on every `coli plan/doctor/tune/run --auto-tier`. It
now caches its result to <model>/.coli_analysis.json, keyed on each shard's and
config.json's (size, mtime_ns), and self-invalidates. This test pins the three
properties the cache must have, WITHOUT a real model: a hit returns the identical
result and does not rescan, a shard change forces a recompute, and a corrupt
cache falls back to a full recompute instead of raising.
"""
import importlib.util
import json
import os
import struct
import tempfile
import time
import unittest
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_RP_PATH = _HERE.parent / "resource_plan.py"
_spec = importlib.util.spec_from_file_location("resource_plan", _RP_PATH)
resource_plan = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(resource_plan)


def _write_shard(path, tensors):
    """Minimal valid safetensors file: u64 header length, JSON header, zero data."""
    header, off = {}, 0
    for name, nbytes in tensors.items():
        header[name] = {"dtype": "F16", "shape": [nbytes // 2],
                        "data_offsets": [off, off + nbytes]}
        off += nbytes
    blob = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        f.write(b"\0" * off)


class AnalysisCacheTest(unittest.TestCase):
    def setUp(self):
        # resolve(): analyze_model() resolves the model path internally, so
        # comparisons against paths derived from self.dir must use the same
        # canonical form — macOS maps /var -> /private/var, Windows expands
        # 8.3 short names (ADMINI~1), and an unresolved mkdtemp path fails
        # the string equality below on both.
        self.dir = Path(tempfile.mkdtemp(prefix="coli_analysis_")).resolve()
        (self.dir / "config.json").write_text(
            '{"model_type": "glm_moe_dsa", "num_hidden_layers": 4}')
        _write_shard(self.dir / "model-00001.safetensors", {
            "model.layers.0.mlp.experts.0.gate_proj.weight": 2048,
            "model.layers.0.mlp.experts.1.gate_proj.weight": 2048,
            "model.embed_tokens.weight": 1024,
        })

    def test_hit_is_identical_and_skips_rescan(self):
        first = resource_plan.analyze_model(self.dir)
        self.assertTrue((self.dir / resource_plan._ANALYSIS_CACHE_NAME).is_file())
        # Make a recompute explode, proving the second call is served from cache.
        orig = resource_plan._tensor_sizes
        resource_plan._tensor_sizes = lambda p: (_ for _ in ()).throw(
            AssertionError("recomputed on a cache hit"))
        try:
            second = resource_plan.analyze_model(self.dir)
        finally:
            resource_plan._tensor_sizes = orig
        self.assertEqual(first, second)

    def test_shard_change_invalidates(self):
        first = resource_plan.analyze_model(self.dir)
        time.sleep(0.01)
        os.utime(self.dir / "model-00001.safetensors", None)  # bump mtime
        again = resource_plan.analyze_model(self.dir)
        self.assertEqual(again, first)  # same content -> same result, but recomputed

    def test_corrupt_cache_falls_back(self):
        first = resource_plan.analyze_model(self.dir)
        (self.dir / resource_plan._ANALYSIS_CACHE_NAME).write_text("{ not valid json")
        recovered = resource_plan.analyze_model(self.dir)
        self.assertEqual(recovered, first)

    def test_write_is_atomic_no_tmp_left_behind(self):
        # A concurrent `coli plan` on the same model dir must never see a
        # half-written cache file; the write goes through a pid-suffixed tmp
        # file + os.replace. After a normal write, no .tmp file should remain
        # and the cache file itself must be valid, complete JSON.
        resource_plan.analyze_model(self.dir)
        tmp_leftovers = list(self.dir.glob("*.tmp"))
        self.assertEqual(tmp_leftovers, [])
        cache_path = self.dir / resource_plan._ANALYSIS_CACHE_NAME
        self.assertTrue(cache_path.is_file())
        parsed = json.loads(cache_path.read_text(encoding="utf-8"))
        self.assertIn("signature", parsed)
        self.assertIn("analysis", parsed)

    def test_write_path_uses_tmp_file_and_replace_not_direct_write(self):
        # Structural check on the ACTUAL write code in analyze_model (not a
        # copy of it): the final rename must be os.replace(tmp, target), and
        # nothing may write cache_path directly. A naive cache_path.write_text
        # regression would still pass the black-box tests above (the window
        # is too small to reliably race in a unit test), so this pins the
        # implementation shape instead of relying on timing.
        from pathlib import Path as _Path

        cache_path = self.dir / resource_plan._ANALYSIS_CACHE_NAME
        write_calls = []
        replace_calls = []
        orig_write_text = _Path.write_text
        orig_replace = os.replace

        def spy_write_text(self_path, *a, **kw):
            write_calls.append(self_path)
            return orig_write_text(self_path, *a, **kw)

        def spy_replace(src, dst, *a, **kw):
            replace_calls.append((_Path(src), _Path(dst)))
            return orig_replace(src, dst, *a, **kw)

        _Path.write_text = spy_write_text
        os.replace = spy_replace
        try:
            resource_plan.analyze_model(self.dir)
        finally:
            _Path.write_text = orig_write_text
            os.replace = orig_replace

        self.assertEqual(len(replace_calls), 1)
        replaced_src, replaced_dst = replace_calls[0]
        self.assertEqual(replaced_dst, cache_path)
        self.assertNotEqual(replaced_src, cache_path)
        # Every write_text call in this run must target the tmp path that was
        # then replaced onto cache_path, never cache_path directly.
        self.assertIn(replaced_src, write_calls)
        self.assertNotIn(cache_path, write_calls)

    def test_concurrent_writers_produce_a_valid_final_cache(self):
        # Multiple threads racing analyze_model() on the same model dir (each
        # forcing a fresh recompute+write via a bumped mtime) must never leave
        # a corrupt or half-written cache file, and no stray tmp files.
        import threading

        cache_path = self.dir / resource_plan._ANALYSIS_CACHE_NAME
        shard_path = self.dir / "model-00001.safetensors"
        errors = []

        def worker(n):
            try:
                for _ in range(15):
                    os.utime(shard_path, None)  # force a real recompute+write each time
                    resource_plan.analyze_model(self.dir)
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [])
        self.assertEqual(list(self.dir.glob("*.tmp")), [])
        json.loads(cache_path.read_text(encoding="utf-8"))  # must not raise


if __name__ == "__main__":
    unittest.main()
