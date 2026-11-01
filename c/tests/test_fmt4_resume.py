"""Exercise fmt4 converter resume with actual tiny safetensors files."""
import contextlib
import io
import json
import os
import multiprocessing
import subprocess
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

try:
    import numpy as np
    from safetensors.numpy import load_file, save_file
except ImportError as exc:
    raise unittest.SkipTest(f"numpy/safetensors unavailable: {exc}")

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
import convert_fmt4_to_fmt2 as converter


class Fmt4ResumeTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.source = self.root / "source"
        self.output = self.root / "output"
        self.source.mkdir()
        self.config = {"hidden_size": 128, "q_lora_rank": 2,
                       "qk_nope_head_dim": 0, "qk_rope_head_dim": 0}
        (self.source / "config.json").write_text(json.dumps(self.config))
        self.name = "model.layers.0.self_attn.q_a_proj.weight"
        self.shard = self.source / "out-00001.safetensors"
        self.write_source(0x77)

    def write_source(self, value):
        save_file({self.name: np.full(128, value, dtype=np.uint8),
                   self.name + ".qs": np.array([1, 2, 3, 4], dtype=np.float32)},
                  str(self.shard))

    def run_converter(self):
        argv = ["convert_fmt4_to_fmt2.py", "--indir", str(self.source),
                "--outdir", str(self.output), "--workers", "1"]
        with mock.patch.object(sys, "argv", argv), \
                contextlib.redirect_stdout(io.StringIO()):
            converter.main()

    def assert_current_output(self):
        expected = converter.convert_shard(str(self.shard), self.config)
        actual = load_file(str(self.output / self.shard.name))
        self.assertEqual(set(actual), set(expected))
        for name in expected:
            np.testing.assert_array_equal(actual[name], expected[name])

    def test_unchanged_resume_does_not_hash_shard_payloads(self):
        self.run_converter()
        actual_hash = converter._sha256
        def forbid_payload_reads(path):
            self.assertFalse(str(path).endswith(".safetensors"), str(path))
            return actual_hash(path)
        with mock.patch.object(converter, "_sha256", side_effect=forbid_payload_reads):
            self.run_converter()
        self.assert_current_output()

    def test_changed_metadata_rehashes_without_reconverting_identical_bytes(self):
        self.run_converter()
        output = self.output / self.shard.name
        for path in (self.shard, output):
            info = path.stat()
            os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns + 1000000000))
        actual_hash = converter._sha256
        with mock.patch.object(converter, "_sha256", wraps=actual_hash) as hashed, \
                mock.patch.object(converter, "convert_shard", wraps=converter.convert_shard) as converted:
            self.run_converter()
        payload_reads = [Path(call.args[0]) for call in hashed.call_args_list
                         if str(call.args[0]).endswith(".safetensors")]
        self.assertCountEqual(payload_reads, [self.shard, output])
        self.assertEqual(converted.call_count, 0)
        with mock.patch.object(converter, "_sha256", side_effect=lambda path: (
                self.fail("unchanged shard rehashed") if str(path).endswith(".safetensors")
                else actual_hash(path))):
            self.run_converter()

    def test_rewritten_source_with_restored_mtime_is_reconverted(self):
        self.run_converter()
        previous = self.shard.stat()
        self.write_source(0x99)
        os.utime(self.shard, ns=(previous.st_atime_ns, previous.st_mtime_ns))
        self.run_converter()
        self.assert_current_output()

    def test_source_change_during_hash_does_not_cache_digest(self):
        actual_hash = converter._sha256
        def edit_after_hash(path):
            digest = actual_hash(path)
            if Path(path) == self.shard:
                self.write_source(0x99)
            return digest
        with mock.patch.object(converter, "_sha256", side_effect=edit_after_hash):
            with self.assertRaisesRegex(SystemExit, "changed while hashing"):
                self.run_converter()
        self.assertFalse((self.output / self.shard.name).exists())

    def test_publication_metadata_change_is_cached_for_next_resume(self):
        actual_replace = converter.os.replace
        def rename_with_new_timestamp(src, dst):
            actual_replace(src, dst)
            if str(dst).endswith(".safetensors"):
                info = os.stat(dst)
                os.utime(dst, ns=(info.st_atime_ns, info.st_mtime_ns + 1000000000))
        with mock.patch.object(converter.os, "replace", side_effect=rename_with_new_timestamp):
            self.run_converter()
        actual_hash = converter._sha256
        def forbid_payload_reads(path):
            self.assertFalse(str(path).endswith(".safetensors"), str(path))
            return actual_hash(path)
        with mock.patch.object(converter, "_sha256", side_effect=forbid_payload_reads):
            self.run_converter()
        self.assert_current_output()

    def test_changed_source_same_size_is_reconverted(self):
        self.run_converter()
        original_bytes = self.shard.stat().st_size
        self.write_source(0x99)
        self.assertEqual(self.shard.stat().st_size, original_bytes)
        self.run_converter()
        self.assert_current_output()

    def test_modified_output_same_size_is_reconverted(self):
        self.run_converter()
        output = self.output / self.shard.name
        original_bytes = output.stat().st_size
        payload = load_file(str(output))
        payload[self.name][:] = 0x00
        save_file(payload, str(output))
        self.assertEqual(output.stat().st_size, original_bytes)
        self.run_converter()
        self.assert_current_output()

    def test_changed_mtp_same_size_is_recopied(self):
        mtp = self.source / "out-mtp-00001.safetensors"
        save_file({"model.norm.weight": np.array([1, 2], dtype=np.float32)}, str(mtp))
        self.run_converter()
        original_bytes = mtp.stat().st_size
        save_file({"model.norm.weight": np.array([3, 4], dtype=np.float32)}, str(mtp))
        self.assertEqual(mtp.stat().st_size, original_bytes)
        self.run_converter()
        self.assertEqual((self.output / mtp.name).read_bytes(), mtp.read_bytes())

    def test_source_edit_during_conversion_does_not_publish(self):
        actual_convert = converter.convert_shard

        def edit_after_conversion(path, config):
            result = actual_convert(path, config)
            self.write_source(0x99)
            return result

        with mock.patch.object(converter, "convert_shard", side_effect=edit_after_conversion):
            with self.assertRaisesRegex(SystemExit, "changed during conversion"):
                self.run_converter()
        self.assertFalse((self.output / self.shard.name).exists())

    def test_removed_source_shard_cannot_remain_in_output(self):
        mtp = self.source / "out-mtp-00001.safetensors"
        save_file({"model.norm.weight": np.array([1, 2], dtype=np.float32)}, str(mtp))
        self.run_converter()
        mtp.unlink()
        with self.assertRaisesRegex(SystemExit, "shards absent from this source"):
            self.run_converter()

    def test_legacy_same_size_wrong_output_is_reconverted(self):
        self.output.mkdir()
        wrong = converter.convert_shard(str(self.shard), self.config)
        wrong[self.name][:] = 0x00
        save_file(wrong, str(self.output / self.shard.name))
        self.run_converter()
        self.assert_current_output()

    def test_skipped_source_edit_after_preflight_is_detected(self):
        self.run_converter()
        actual_free = converter.free_gb
        edited = False

        def edit_after_preflight(path):
            nonlocal edited
            free = actual_free(path)
            if not edited:
                self.write_source(0x99)
                edited = True
            return free

        with mock.patch.object(converter, "free_gb", side_effect=edit_after_preflight):
            with self.assertRaisesRegex(SystemExit, "changed during conversion"):
                self.run_converter()

    def test_added_source_shard_during_conversion_is_detected(self):
        actual_convert = converter.convert_shard

        def add_after_conversion(path, config):
            result = actual_convert(path, config)
            save_file({"model.norm.weight": np.array([1, 2], dtype=np.float32)},
                      str(self.source / "out-new.safetensors"))
            return result

        with mock.patch.object(converter, "convert_shard", side_effect=add_after_conversion):
            with self.assertRaisesRegex(SystemExit, "source shard membership changed"):
                self.run_converter()

    def test_added_output_shard_during_conversion_is_detected(self):
        actual_convert = converter.convert_shard

        def add_after_conversion(path, config):
            result = actual_convert(path, config)
            save_file({"model.norm.weight": np.array([1, 2], dtype=np.float32)},
                      str(self.output / "out-new.safetensors"))
            return result

        with mock.patch.object(converter, "convert_shard", side_effect=add_after_conversion):
            with self.assertRaisesRegex(SystemExit, "output shard membership changed"):
                self.run_converter()

    def test_receipt_rejects_output_modified_after_atomic_publication(self):
        actual_replace = converter.os.replace
        edited = False

        def edit_after_publication(src, dst):
            nonlocal edited
            actual_replace(src, dst)
            if str(dst).endswith(".safetensors") and not edited:
                payload = load_file(str(dst))
                payload[self.name][:] = 0x00
                save_file(payload, str(dst))
                edited = True

        with mock.patch.object(converter.os, "replace", side_effect=edit_after_publication):
            with self.assertRaisesRegex(SystemExit, "output changed during conversion"):
                self.run_converter()

    def cli(self):
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OPENBLAS_NUM_THREADS="1")
        return subprocess.run(
            [sys.executable, str(TOOLS / "convert_fmt4_to_fmt2.py"),
             "--indir", str(self.source), "--outdir", str(self.output), "--workers", "2"],
            env=env, capture_output=True, text=True, timeout=30)

    def test_cli_two_workers_matches_serial_conversion(self):
        self.config["kv_lora_rank"] = 2
        (self.source / "config.json").write_text(json.dumps(self.config))
        other = "model.layers.0.self_attn.kv_a_proj_with_mqa.weight"
        second = self.source / "out-00002.safetensors"
        save_file({other: np.full(128, 0x99, dtype=np.uint8),
                   other + ".qs": np.array([1, 2, 3, 4], dtype=np.float32)}, str(second))
        result = self.cli()
        self.assertEqual(result.returncode, 0, result.stderr)
        for shard in (self.shard, second):
            expected = converter.convert_shard(str(shard), self.config)
            actual = load_file(str(self.output / shard.name))
            for name in expected:
                np.testing.assert_array_equal(actual[name], expected[name])
        resumed = self.cli()
        self.assertEqual(resumed.returncode, 0, resumed.stderr)
        self.assertIn("[RESUME]", resumed.stdout)

    def test_cli_worker_failure_returns_without_hanging(self):
        # Header-only planning succeeds, then the real worker's safetensors
        # reader rejects a truncated tensor payload. No process is simulated.
        self.shard.write_bytes(self.shard.read_bytes()[:-1])
        result = self.cli()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("SafetensorError", result.stderr)
        self.assertFalse((self.output / "config.json").exists())

    def test_simulated_legacy_replacement_budget_is_per_shard(self):
        # The reported disk budget is simulated. The actual filesystem has
        # ordinary free space, and both real outputs are only about 70 KiB.
        self.config.update(q_lora_rank=1024, kv_lora_rank=1024)
        (self.source / "config.json").write_text(json.dumps(self.config))
        other = "model.layers.0.self_attn.kv_a_proj_with_mqa.weight"
        second = self.source / "out-00002.safetensors"
        self.output.mkdir()
        sizes = []
        for shard, name in ((self.shard, self.name), (second, other)):
            save_file({name: np.full(65536, 0x99, dtype=np.uint8),
                       name + ".qs": np.tile(np.array([1, 2], dtype=np.float32), 1024)}, str(shard))
            stale = converter.convert_shard(str(shard), self.config)
            stale[name][:] = 0
            save_file(stale, str(self.output / shard.name))
            sizes.append((self.output / shard.name).stat().st_size)
        simulated_free = max(sizes) + 16384  # includes a margin for tiny metadata
        self.assertLess(simulated_free, sum(sizes))
        actual_lstat = converter.os.lstat

        def simulated_allocation(path, *args, **kwargs):
            info = actual_lstat(path, *args, **kwargs)
            return types.SimpleNamespace(st_mode=info.st_mode, st_nlink=1,
                                         st_size=info.st_size,
                                         st_blocks=(info.st_size + 511) // 512)

        # Simulate ordinary singly linked, allocated destination files on every
        # host; this is a planning control, not Windows/disk-space qualification.
        with mock.patch.object(converter, "free_gb", return_value=simulated_free / 1e9), \
                mock.patch.object(converter.os, "lstat", side_effect=simulated_allocation):
            self.run_converter()
        for shard in (self.shard, second):
            expected = converter.convert_shard(str(shard), self.config)
            actual = load_file(str(self.output / shard.name))
            for name in expected:
                np.testing.assert_array_equal(actual[name], expected[name])

    def test_simulated_budget_below_one_temp_shard_is_rejected(self):
        with mock.patch.object(converter, "free_gb", return_value=1 / 1e9):
            with self.assertRaisesRegex(SystemExit, "not enough free space"):
                self.run_converter()
        self.assertFalse((self.output / self.shard.name).exists())

    def test_worker_validation_exit_is_delivered_to_parent(self):
        save_file({self.name: np.zeros(128, dtype=np.uint8)}, str(self.shard))
        # An actual spawned worker must deliver its validation failure through
        # the result channel, rather than exit and leave the parent waiting.
        with multiprocessing.get_context("spawn").Pool(2) as pool:
            result = pool.apply_async(converter._convert_one,
                                      ((0, str(self.shard), self.config),))
            with self.assertRaisesRegex(RuntimeError, "no .qs"):
                result.get(timeout=5)

    def test_unchanged_shards_resume_without_conversion(self):
        self.run_converter()
        output = self.output / self.shard.name
        original = output.read_bytes()
        with mock.patch.object(converter, "convert_shard", wraps=converter.convert_shard) as convert:
            self.run_converter()
        self.assertEqual(convert.call_count, 0)
        self.assertEqual(output.read_bytes(), original)
        self.assert_current_output()


if __name__ == "__main__":
    unittest.main()
