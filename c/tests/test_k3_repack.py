"""Kimi K3 repack resume and cumulative-index regressions."""
import importlib
import importlib.util
import io
import json
import os
import struct
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock


TOOLS = Path(__file__).resolve().parent.parent / "tools"


def load_tool():
    """The index path is stdlib-only; stub numpy when optional tooling is absent."""
    try:
        importlib.import_module("numpy")
        stubbed = False
    except ImportError:
        stubbed = True
    if stubbed:
        sys.modules["numpy"] = types.ModuleType("numpy")
    try:
        spec = importlib.util.spec_from_file_location("k3_repack_under_test", TOOLS / "k3_repack.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if stubbed:
            sys.modules.pop("numpy", None)


k3_repack = load_tool()


def write_shard(path, tensors):
    offset = 0
    header = {}
    payload = b""
    for name, data in tensors:
        header[name] = {"dtype": "U8", "shape": [len(data)],
                        "data_offsets": [offset, offset + len(data)]}
        payload += data
        offset += len(data)
    raw = json.dumps(header, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)


class K3RepackIndexTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dst = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def shard(self, number, tensors):
        path = self.dst / f"model-{number:05d}-of-000094.safetensors"
        write_shard(path, tensors)
        return path

    def index(self):
        return json.loads((self.dst / "model.safetensors.index.json").read_text())

    def test_rebuild_combines_disjoint_partial_runs_and_counts_existing_payloads(self):
        first = self.shard(1, [("layer.1.weight", b"1234")])
        self.assertEqual(k3_repack.rebuild_index(self.dst), (1, 4))
        self.assertEqual(self.index(), {
            "metadata": {"total_size": 4},
            "weight_map": {"layer.1.weight": first.name},
        })

        second = self.shard(2, [("layer.2.weight", b"abcdef")])
        self.assertEqual(k3_repack.rebuild_index(self.dst), (2, 10))
        self.assertEqual(self.index(), {
            "metadata": {"total_size": 10},
            "weight_map": {
                "layer.1.weight": first.name,
                "layer.2.weight": second.name,
            },
        })

        before = (self.dst / "model.safetensors.index.json").read_bytes()
        self.assertEqual(k3_repack.rebuild_index(self.dst), (2, 10))
        self.assertEqual((self.dst / "model.safetensors.index.json").read_bytes(), before)

    def test_duplicate_tensor_aborts_without_replacing_the_previous_index(self):
        self.shard(1, [("duplicate.weight", b"1234")])
        k3_repack.rebuild_index(self.dst)
        index_path = self.dst / "model.safetensors.index.json"
        previous = index_path.read_bytes()
        self.shard(2, [("duplicate.weight", b"5678")])

        with self.assertRaisesRegex(ValueError, "duplicate tensor.*model-00001.*model-00002"):
            k3_repack.rebuild_index(self.dst)

        self.assertEqual(index_path.read_bytes(), previous)
        self.assertFalse((self.dst / "model.safetensors.index.json.tmp").exists())

    def test_rebuilt_index_passes_doctor_container_validation(self):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        try:
            from doctor import deep_container_report
        finally:
            sys.path.pop(0)
        tensors = [
            ("model.embed_tokens.weight", b"e"),
            ("model.norm.weight", b"n"),
        ]
        self.shard(1, tensors)
        self.shard(2, [("lm_head.weight", b"head")])
        k3_repack.rebuild_index(self.dst)

        report = deep_container_report(self.dst)

        self.assertEqual(report["index"]["status"], "pass")
        self.assertEqual(report["required"]["status"], "pass")

    def test_index_is_flushed_before_atomic_replace(self):
        self.shard(1, [("layer.weight", b"1234")])
        calls = []
        real_replace = os.replace

        def replace(src, dst):
            calls.append((Path(src), Path(dst)))
            real_replace(src, dst)

        with mock.patch.object(k3_repack.os, "fsync") as fsync, \
             mock.patch.object(k3_repack.os, "replace", side_effect=replace):
            k3_repack.rebuild_index(self.dst)

        self.assertEqual(fsync.call_count, 1)
        self.assertEqual(calls, [(
            self.dst / "model.safetensors.index.json.tmp",
            self.dst / "model.safetensors.index.json",
        )])


class K3RepackResumeIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.src = self.root / "src"
        self.dst = self.root / "dst"
        self.src.mkdir()
        (self.src / "config.json").write_text(json.dumps({
            "text_config": {"linear_attn_config": {"kda_layers": [1]}},
        }))
        for number in (1, 2):
            write_shard(
                self.src / f"model-{number:05d}-of-000096.safetensors",
                [(f"model.layers.{number}.input_layernorm.weight", bytes([number]) * number)],
            )

    def tearDown(self):
        self.tmp.cleanup()

    def run_main(self, shard):
        argv = ["k3_repack.py", str(self.src), str(self.dst), "--shards", str(shard)]
        with mock.patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()):
            k3_repack.main()

    def test_disjoint_shard_passes_publish_one_cumulative_index(self):
        self.run_main(1)
        self.run_main(2)
        index_path = self.dst / "model.safetensors.index.json"
        document = json.loads(index_path.read_text())

        self.assertEqual(document, {
            "metadata": {"total_size": 3},
            "weight_map": {
                "model.layers.1.input_layernorm.weight":
                    "model-00001-of-000094.safetensors",
                "model.layers.2.input_layernorm.weight":
                    "model-00002-of-000094.safetensors",
            },
        })

        before = index_path.read_bytes()
        self.run_main(1)
        self.assertEqual(index_path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
