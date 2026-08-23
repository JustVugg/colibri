import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.mirror_plan import (RECEIPT, MirrorError, create_plan, discover_shards,
                               stage_mirror, usage_counts, verify_mirror)


class MirrorPlannerTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        # .resolve(): on macOS /var is a symlink to /private/var, so mkdtemp hands
        # back /var/folders/... while mirror_plan resolves every path it is given
        # (discover_shards, create_plan). Comparing a resolved path from the tool
        # against an unresolved one built here fails on macOS and passes on Linux.
        self.root = Path(self.temporary.name).resolve()
        self.model = self.root / "model"
        self.mirror = self.root / "mirror"
        self.split = self.root / "split"
        self.model.mkdir()
        self.split.mkdir()
        self.usage = self.model / ".coli_usage"

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def write_shard(directory, name, tensors):
        offset = 0
        header = {}
        payload = bytearray()
        for tensor, size in tensors:
            header[tensor] = {"dtype": "U8", "shape": [size],
                              "data_offsets": [offset, offset + size]}
            payload.extend(bytes([len(header) % 251]) * size)
            offset += size
        encoded = json.dumps(header, separators=(",", ":")).encode()
        path = directory / name
        path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
        return path

    def test_usage_parser_ignores_malformed_rows_and_accumulates(self):
        self.usage.write_text("0 1 7\ninvalid\n0 1 5\n-1 2 9\n1 2 0\n", encoding="utf-8")
        self.assertEqual(usage_counts(self.usage), {(0, 1): 12})

    def test_plan_activates_hot_gate_shard_before_companion_shard(self):
        gate = self.write_shard(self.model, "hot-gate.safetensors", [
            ("model.layers.0.mlp.experts.0.gate_proj.weight", 20),
        ])
        companion = self.write_shard(self.model, "hot-down.safetensors", [
            ("model.layers.0.mlp.experts.0.down_proj.weight", 60),
        ])
        self.write_shard(self.model, "cold-gate.safetensors", [
            ("model.layers.0.mlp.experts.1.gate_proj.weight", 20),
        ])
        self.usage.write_text("0 0 100\n0 1 1\n", encoding="utf-8")
        budget = gate.stat().st_size + companion.stat().st_size

        plan, selected = create_plan(self.model, self.mirror, [], self.usage, budget, 0)

        self.assertTrue(plan["admitted"])
        self.assertEqual([item["name"] for item in selected],
                         ["hot-gate.safetensors", "hot-down.safetensors"])
        self.assertEqual(selected[0]["activated_experts"], 1)
        self.assertEqual(selected[1]["activated_experts"], 0)

    def test_plan_recognizes_k3_expert_tensors_and_packed_w1_anchor(self):
        prefix = "language_model.model.layers.1.block_sparse_moe.experts.7"
        anchor = self.write_shard(self.model, "k3-w1.safetensors", [
            (f"{prefix}.w1.weight_packed", 11),
        ])
        companion = self.write_shard(self.model, "k3-rest.safetensors", [
            (f"{prefix}.w1.weight_scale", 2),
            (f"{prefix}.w2.weight_packed", 13),
            (f"{prefix}.w2.weight_scale", 3),
            (f"{prefix}.w3.weight_packed", 17),
            (f"{prefix}.w3.weight_scale", 5),
        ])
        self.usage.write_text("1 7 99\n", encoding="utf-8")
        budget = anchor.stat().st_size + companion.stat().st_size

        plan, selected = create_plan(self.model, self.mirror, [], self.usage, budget, 0)

        self.assertTrue(plan["admitted"])
        self.assertEqual([item["name"] for item in selected],
                         ["k3-w1.safetensors", "k3-rest.safetensors"])
        self.assertEqual(selected[0]["weighted_expert_bytes"], 99 * 11)
        self.assertEqual(selected[1]["weighted_expert_bytes"], 99 * 40)
        self.assertEqual([item["activated_experts"] for item in selected], [1, 0])

    def test_k3_names_accept_engine_prefixes_and_reject_near_matches(self):
        accepted = [
            ("model.layers.2.block_sparse_moe.experts.8.w1.weight_packed", 7),
            ("language_model.model.layers.3.block_sparse_moe.experts.9."
             "w1.weight_packed", 11),
        ]
        rejected = [
            ("model.layers.2.block_sparse_moe.shared_experts.w1.weight_packed", 13),
            ("model.layers.2.block_sparse_moe.experts.8.w4.weight_packed", 17),
            ("model.layers.2.block_sparse_moe.experts.8.w1.weight_scale_inv", 19),
            ("foo.model.layers.2.block_sparse_moe.experts.8.w1.weight_packed", 23),
        ]
        self.write_shard(self.model, "k3-names.safetensors", accepted + rejected)

        _directories, candidates = discover_shards(self.model, [])

        self.assertEqual(candidates[0]["expert_bytes"], {(2, 8): 7, (3, 9): 11})
        self.assertEqual(candidates[0]["anchor_experts"], {(2, 8), (3, 9)})

    def test_plan_requires_learned_usage_instead_of_guessing(self):
        shard = self.write_shard(self.model, "model.safetensors", [
            ("model.layers.0.mlp.experts.0.gate_proj.weight", 8),
        ])
        plan, selected = create_plan(
            self.model, self.mirror, [], self.usage, shard.stat().st_size, 0)
        self.assertFalse(plan["admitted"])
        self.assertEqual(plan["reason"], "usage_history_missing")
        self.assertEqual(selected, [])

    def test_split_directories_are_searched_and_basenames_are_deduplicated(self):
        primary = self.write_shard(self.model, "same.safetensors", [
            ("model.layers.0.mlp.experts.0.gate_proj.weight", 8),
        ])
        self.write_shard(self.split, "same.safetensors", [
            ("model.layers.0.mlp.experts.1.gate_proj.weight", 16),
        ])
        extra = self.write_shard(self.split, "extra.safetensors", [
            ("model.layers.0.mlp.experts.2.gate_proj.weight", 12),
        ])
        _directories, candidates = discover_shards(self.model, [self.split])
        by_name = {item["name"]: item for item in candidates}
        self.assertEqual(by_name["same.safetensors"]["source"], primary)
        self.assertEqual(by_name["extra.safetensors"]["source"], extra)

    def test_stage_is_atomic_resumable_and_sha256_verified(self):
        source = self.write_shard(self.model, "hot.safetensors", [
            ("model.layers.0.mlp.experts.0.gate_proj.weight", 32),
        ])
        self.usage.write_text("0 0 25\n", encoding="utf-8")
        budget = source.stat().st_size

        result = stage_mirror(self.model, self.mirror, [], self.usage, budget, 0)

        self.assertTrue(result["ready"])
        target = self.mirror / source.name
        self.assertEqual(target.read_bytes(), source.read_bytes())
        self.assertTrue((self.mirror / RECEIPT).is_file())
        first_mtime = target.stat().st_mtime_ns

        repeated = stage_mirror(self.model, self.mirror, [], self.usage, budget, 0)
        self.assertTrue(repeated["ready"])
        self.assertEqual(repeated["plan"]["remaining_copy_bytes"], 0)
        self.assertEqual(target.stat().st_mtime_ns, first_mtime)

        content = target.read_bytes()
        target.write_bytes(bytes([content[0] ^ 1]) + content[1:])
        verification = verify_mirror(self.mirror)
        self.assertFalse(verification["ready"])
        self.assertEqual(verification["failures"], ["hot.safetensors (sha256)"])

    def test_reserve_preflight_writes_no_shard_or_receipt(self):
        source = self.write_shard(self.model, "hot.safetensors", [
            ("model.layers.0.mlp.experts.0.gate_proj.weight", 32),
        ])
        self.usage.write_text("0 0 25\n", encoding="utf-8")
        disk = mock.Mock(free=source.stat().st_size - 1)
        with mock.patch("tools.mirror_plan.shutil.disk_usage", return_value=disk):
            result = stage_mirror(
                self.model, self.mirror, [], self.usage, source.stat().st_size, 0)
        self.assertFalse(result["admitted"])
        self.assertEqual(result["reason"], "free_space_reserve")
        self.assertFalse((self.mirror / source.name).exists())
        self.assertFalse((self.mirror / RECEIPT).exists())

    def test_verify_rejects_receipt_path_traversal(self):
        self.mirror.mkdir()
        receipt = {"schema": "colibri.partial-mirror.v1", "files": [{
            "name": "../outside.safetensors", "size": 1, "sha256": "0" * 64,
        }]}
        (self.mirror / RECEIPT).write_text(json.dumps(receipt), encoding="utf-8")
        result = verify_mirror(self.mirror)
        self.assertFalse(result["ready"])
        self.assertEqual(result["failures"], ["invalid_receipt_entry"])

    def test_invalid_safetensors_header_fails_closed(self):
        (self.model / "bad.safetensors").write_bytes(struct.pack("<Q", 999) + b"{}")
        with self.assertRaises(MirrorError):
            discover_shards(self.model, [])


if __name__ == "__main__":
    unittest.main()
