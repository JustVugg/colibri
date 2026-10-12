#!/usr/bin/env python3
"""The SERVE prefix snapshot is in the planner budget.

kolibri1 SERVE takes a second copy of the KV state after every completed prompt
(k1_snap_take: the full layers' history [0, P), the whole ring of the sliding layers,
the int64 ring positions) and keeps it for prefix reuse.  Its worst case is one more KV
bank, 2 * kv_heads * head_dim * 4 B * (10 * L + 40 * 513) for the release (about 10.08 GiB at
L = 262,144).  Design decision: the planner CHARGES it (the geometry counts live state +
snapshot), so the expert cache gives way and the generic reserve stays intact; the engine's
snapshot is never refused or bounded.  What this test pins:

  * the geometry of the real 50-layer release at 262,144 tokens (live + snapshot + metadata);
  * build_plan() over the tiny fixture at 262,144 with a tight RAM budget: dense + runtime
    + expert cache never exceeds the budget (the 12% reserve of the automatic path included),
    and the snapshot takes the expert cache down where, uncharged, it would have fit.

Needs numpy for the fixture (tools/make_kolibri1_tiny.py) in the build_plan part.
"""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import family_registry as fr  # noqa: E402
import resource_plan as rp  # noqa: E402

try:
    import numpy  # noqa: F401
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False

GIB = 1 << 30
TOOL = ROOT / "tools" / "make_kolibri1_tiny.py"


def release_config():
    return {
        "model_type": "kolibri1", "hidden_size": 2560, "num_hidden_layers": 50,
        "num_attention_heads": 48, "num_key_value_heads": 4, "head_dim": 128,
        "vocab_size": 128000, "num_experts": 384, "num_experts_per_tok": 6,
        "moe_intermediate_size": 512, "sliding_window": 513,
        "layer_types": ["full_attention" if l % 5 == 4 else "sliding_attention" for l in range(50)],
    }


def geometry_of(config, context):
    by_id, _ = fr._build_registry(fr.FAMILIES)
    resolved = type("R", (), {"descriptor": by_id["kolibri1"], "family_config": config,
                              "model_dir": "."})()
    return fr.planner_geometry(resolved, context)


class Kolibri1SnapshotGeometryTest(unittest.TestCase):
    def test_release_at_262144_charges_live_state_and_the_snapshot(self):
        ctx = 262144
        g = geometry_of(release_config(), ctx)
        row = 2 * 4 * 128 * 4                               # K + V, 4 KV heads, head_dim 128, f32
        bank = row * (10 * ctx + 40 * 513)                  # one KV bank
        self.assertAlmostEqual(bank / GIB, 10.08, places=2)
        full, ring = row * 10 * ctx, row * 40 * 513
        ring_pos = 40 * 513 * 8                             # int64 position per ring row, live and snapshot
        total = g.context_state_bytes + g.fixed_state_bytes
        self.assertGreaterEqual(total, 2 * bank + 2 * ring_pos)   # live + snapshot, plus the metadata
        self.assertLess(total, 2 * bank + 2 * ring_pos + 4 * ctx * 4 + 64 * 1024)   # and nothing else of size
        self.assertGreaterEqual(g.context_state_bytes, 2 * full)
        self.assertGreaterEqual(g.fixed_state_bytes, 2 * ring + 2 * ring_pos)

    def test_short_context_cost_scales_with_the_prompt(self):
        small = geometry_of(release_config(), 8192)
        row = 2 * 4 * 128 * 4
        self.assertGreaterEqual(small.context_state_bytes + small.fixed_state_bytes,
                                2 * row * (10 * 8192 + 40 * 513))
        self.assertAlmostEqual(row * (10 * 8192 + 40 * 513) / GIB, 0.39, places=2)


@unittest.skipUnless(HAVE_NUMPY, "numpy is needed to generate the tiny fixture")
class Kolibri1SnapshotPlanTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="k1-snapbudget-")
        cls.model = Path(cls.tmp.name) / "k1"
        subprocess.run([sys.executable, str(TOOL), "--out", str(cls.model), "--layout", "release"],
                       check=True, capture_output=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def plan(self, ram_gb=0, available=None, ctx=262144):
        return rp.build_plan(str(self.model), ram_gb=ram_gb, context=ctx,
                             available_memory=available, available_disk=10**12, gpus=[],
                             physical_cpus=2, cpu_sockets=1)["tiers"]["ram"]

    def snapshot_bytes(self, ctx):
        # tiny fixture: 6 layers (1 full), 2 KV heads, head_dim 128, window 17; independent of the planner
        row = 2 * 2 * 128 * 4
        return row * (1 * ctx + 5 * 17) + 5 * 17 * 8 + 3 * 6 * 8 + 64 + ctx * 4

    def test_runtime_reservation_carries_the_snapshot(self):
        ctx = 262144
        ram = self.plan(ram_gb=512, ctx=ctx)
        tiny = self.plan(ram_gb=512, ctx=64)
        live = ram["sequence_state_bytes"] + ram["fixed_state_bytes"]
        self.assertGreaterEqual(live, 2 * 2 * 2 * 128 * 4 * ctx)                  # live full layer + snapshot copy
        self.assertGreaterEqual(live - (tiny["sequence_state_bytes"] + tiny["fixed_state_bytes"]),
                                self.snapshot_bytes(ctx) - self.snapshot_bytes(64) - 64 * 1024)

    def test_tight_budget_charges_the_cache_not_the_reserve(self):
        ctx = 262144
        probe = self.plan(ram_gb=512, ctx=ctx)
        snap = self.snapshot_bytes(ctx)
        runtime_uncharged = probe["runtime_bytes"] - snap       # what the planner counted before it charged the snapshot
        dense = probe["dense_bytes"]
        per_cap = rp.qwen38_int4_sidecar(rp.analyze_model(str(self.model)))["per_cap_bytes"]
        # a budget the old accounting called fine (>= 3 expert slots per layer beyond the fixed
        # footprint) and the charged one cannot honour: the snapshot is bigger than that room
        room = 3 * per_cap
        self.assertLess(room, snap)
        budget = dense + runtime_uncharged + room
        self.assertGreaterEqual((budget - dense - runtime_uncharged) // per_cap, 3)   # uncharged: it would have planned >= 3 slots
        ram = self.plan(ram_gb=budget / rp.GB, ctx=ctx)
        self.assertEqual(ram["budget_bytes"], int(budget / rp.GB * rp.GB))
        self.assertEqual(ram["expert_cache_bytes"], 0)          # the reserve is not spent on the snapshot
        self.assertEqual(ram["cache_slots_per_layer"], 0)
        # SERVE start-up refuses instead of launching a plan that cannot hold the snapshot
        import openai_server as gw
        env = {"RAM_GB": str(budget / rp.GB), fr.family_by_id("kolibri1").limits.context_env: str(ctx)}
        with self.assertRaisesRegex(ValueError, "cannot hold one expert slot"):
            gw.cap_for_arch("kolibri1", None, env, str(self.model))
        # with room for the snapshot AND the slots, the cache is exactly what is left
        roomy = dense + probe["runtime_bytes"] + 3 * per_cap
        ram = self.plan(ram_gb=roomy / rp.GB, ctx=ctx)
        self.assertGreaterEqual(ram["cache_slots_per_layer"], 3)
        self.assertLessEqual(dense + ram["runtime_bytes"] + ram["expert_cache_bytes"], ram["budget_bytes"])

    def test_automatic_budget_keeps_the_twelve_percent_reserve(self):
        ctx = 262144
        probe = self.plan(ram_gb=512, ctx=ctx)
        available = int((probe["dense_bytes"] + probe["runtime_bytes"]) / 0.88) + 3 * GIB
        ram = self.plan(available=available, ctx=ctx)
        self.assertEqual(ram["budget_bytes"], int(available * 0.88))
        self.assertLessEqual(ram["dense_bytes"] + ram["runtime_bytes"] + ram["expert_cache_bytes"],
                             int(available * 0.88))
        self.assertLessEqual(int(available * 0.88), available - int(available * 0.12) + 1)


if __name__ == "__main__":
    unittest.main()
