"""The packed cache's planner re-price (family_registry._dsv41_geometry): the host's
KV state priced as the cache's packed code rows while COLI_VK_KV_FP32 is unset (the
default now -- unset runs packed, mirrors and host alike), fp32 under =1 (the exact
opt-out, which takes the device mirrors with it).

What is pinned here is the SEAM -- the per-row width swap and its guard, at the released
geometry's config and the three context lengths whose slot deltas the memory model
predicted (0/+1/+2 at 64K/128K/256K against per_cap 771.14 MiB; 16K is 0 too). The slot
arithmetic itself (cache_bytes // per_cap) needs the checkpoint's own tensors for
per_cap, so its end-to-end evidence is the serve smoke's "[v41] RAM plan" line, the
packed default against FP32=1 -- the same line the ladder reproduced for every measured
context.
"""
import importlib.util
import os
import unittest
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SPEC = importlib.util.spec_from_file_location("family_registry", _HERE.parent / "family_registry.py")
fr = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(fr)

# The released V4.1 Flash text config, the fields the geometry reads: 40 layers, 4 KV
# sources at ratios 2/2/2/1 (sum 1/r = 2.5, so a position costs 2.5 compressed rows),
# window 128, head_dim 512, index_head_dim 128, 3 DSpark stages of a 5-token block. The
# stage count is spelled the way the checkpoint spells it -- num_nextn_predict_layers,
# NOT n_mtp_layers, which the released config.json does not carry at all -- and
# compress_ratios runs to the checkpoint's own length, 43 (the entries past the layers
# are 0 and unread).
def _config(drop=(), **over):
    ratios = [0] * 43
    for layer, ratio in ((2, 2), (8, 2), (14, 2), (20, 1)):
        ratios[layer] = ratio
    cfg = {
        "num_hidden_layers": 40,
        "n_routed_experts": 384,
        "hidden_size": 2048,
        "head_dim": 512,
        "window_size": 128,
        "index_head_dim": 128,
        "hc_mult": 4,
        "compress_ratios": ratios,
        "kv_source_layer_ids": [2, 8, 14, 20],
        "num_nextn_predict_layers": 3,
        "dspark_block_size": 5,
        "dspark_target_layer_ids": [37, 38, 39],
        "moe_inter_dim": 2048,
    }
    for key in drop:
        cfg.pop(key, None)
    cfg.update(over)
    return {"text_config": cfg}


def _geometry(context, fp32, drop=(), **over):
    """The planner's geometry at a flag setting. fp32 True: COLI_VK_KV_FP32=1 (the
    exact opt-out); False: the var UNSET -- the default, packed."""
    env = dict(os.environ)
    env.pop("COLI_VK_KV_FP32", None)
    if fp32:
        env["COLI_VK_KV_FP32"] = "1"
    old = os.environ
    try:
        os.environ = env
        return fr._dsv41_geometry(_config(drop=drop, **over), context, None)
    finally:
        os.environ = old


def _state(context, fp32, drop=(), **over):
    return _geometry(context, fp32, drop, **over).context_state_bytes


# The fixed term at the released geometry, which is the engine's own v41_host_kv_bytes:
# 43 window rings (40 layers + 3 DSpark stages) plus the rollback rows a speculative
# verify parks, spec_block + 1 = 6 a layer, at the row the host holds -- 528 B packed,
# 2048 B fp32.
_FIXED_ROWS = 43 * 128 + 40 * 6


class Dsv41KvqHostPlanTest(unittest.TestCase):
    def test_the_packed_price_the_fp32_one_and_the_default(self):
        # a position costs 2.5 compressed rows: 6400 B fp32, 890 B packed -- and the
        # var unset IS the packed price (the engine's default, mirrored here)
        for ctx in (65536, 131072, 262144):
            self.assertEqual(_state(ctx, True), 6400 * ctx, f"fp32 state at ctx {ctx}")
            self.assertEqual(_state(ctx, False), 890 * ctx, f"packed state at ctx {ctx}")

    def test_the_window_rings_are_priced_at_the_row_the_host_holds(self):
        # The rings are (layers + stages) * window rows plus the rollback rows a layer,
        # all at the HOST's own row width, so the packed default's 528-byte row is the
        # fixed term's price too: (43 * 128 + 40 * 6) * 528 = 3 032 832 B against the
        # fp32 layout's * 2048 = 11 763 712 B -- the 8 730 880 B (8.33 MiB) the old
        # fixed term over-reserved for a host that holds codes. At any other geometry
        # the guard declines and the rings are fp32.
        self.assertEqual(_geometry(0, True).fixed_state_bytes, _FIXED_ROWS * 2048)
        self.assertEqual(_geometry(0, False).fixed_state_bytes, _FIXED_ROWS * 528)
        self.assertEqual(_geometry(0, True).fixed_state_bytes - _geometry(0, False).fixed_state_bytes, 8730880)
        self.assertEqual(_geometry(0, False, head_dim=64, index_head_dim=32).fixed_state_bytes,
                         _FIXED_ROWS * 64 * 4)

    def test_the_released_shape_prices_the_engines_host_kv_total(self):
        # The engine's own closed form (deepseek_v41.c v41_host_kv_bytes) at 16K: the
        # 2.5 compressed rows a position plus the fixed term, which is the number the
        # serve smoke's ledger prints ("host kv 16.8 MiB (packed)"). Before the stage
        # and rollback terms were read from this config shape the planner said
        # 17 285 120 B -- 3 stage rings and 240 rollback rows short.
        packed = _geometry(16384, False)
        self.assertEqual(packed.context_state_bytes, 890 * 16384)
        self.assertEqual(packed.fixed_state_bytes, 3032832)
        self.assertEqual(packed.context_state_bytes + packed.fixed_state_bytes, 17614592)
        fp32 = _geometry(16384, True)
        self.assertEqual(fp32.context_state_bytes, 6400 * 16384)
        self.assertEqual(fp32.fixed_state_bytes, 11763712)
        self.assertEqual(fp32.context_state_bytes + fp32.fixed_state_bytes, 116621312)

    def test_a_config_spelling_the_stages_n_mtp_layers_prices_the_same(self):
        # backward compatibility: the vendor's inference/config.json (and every fixture
        # this repository wrote before) spells the count n_mtp_layers. Either key is
        # the same three stages, and the explicit key wins when both are present.
        legacy = _geometry(0, False, drop=("num_nextn_predict_layers",), n_mtp_layers=3)
        self.assertEqual(legacy.fixed_state_bytes, 3032832)
        both = _geometry(0, False, n_mtp_layers=3, num_nextn_predict_layers=1)
        self.assertEqual(both.fixed_state_bytes, 3032832)

    def test_a_stage_key_without_a_draft_block_prices_no_stage_rings(self):
        # A stage key with no dspark_block_size: the engine zeroes n_mtp before it loads
        # a head, so the planner must not price stage rings. The rollback rows stay: a
        # layer allocates them whether or not anything drafts.
        no_block = _geometry(0, False, drop=("dspark_block_size",))
        self.assertEqual(no_block.fixed_state_bytes, 40 * 128 * 528 + 40 * 1 * 528)
        zero_block = _geometry(0, False, dspark_block_size=0)
        self.assertEqual(zero_block.fixed_state_bytes, 40 * 128 * 528 + 40 * 1 * 528)
        # Same gate on the target layers: a block but nothing to draft at keeps no rings.
        no_targets = _geometry(0, False, drop=("dspark_target_layer_ids",))
        self.assertEqual(no_targets.fixed_state_bytes, 40 * 128 * 528 + 40 * 6 * 528)

    def test_a_config_with_no_stage_key_prices_the_rollback_rows_alone(self):
        # No stage key at all: 0 stages -- and still one rollback row a layer, not none.
        # spec_block 0 is the engine's `rows = 1` branch, the same row count model_load
        # allocates when there is no block to park.
        bare = _geometry(0, False, drop=("num_nextn_predict_layers", "dspark_block_size"))
        self.assertEqual(bare.fixed_state_bytes, 40 * 128 * 528 + 40 * 1 * 528)
        self.assertEqual(_geometry(0, False, drop=("num_nextn_predict_layers",)).fixed_state_bytes,
                         40 * 128 * 528 + 40 * 6 * 528)

    def test_a_stage_or_block_count_that_is_not_a_count_is_refused(self):
        for over in ({"num_nextn_predict_layers": -1}, {"num_nextn_predict_layers": "3"},
                     {"dspark_block_size": -1}, {"dspark_block_size": True}):
            with self.assertRaises(ValueError, msg=f"config {over}"):
                _geometry(0, False, **over)

    def test_the_freed_ram_against_the_slot_it_buys(self):
        # 5510 B a position freed. The slot a freed budget BUYS is the floor of the whole
        # cache_bytes / per_cap, not the floor of the freed bytes alone: the ladder's
        # 0/+1/+2 at 64K/128K/256K (per_cap 771.14 MiB, one expert of every layer) comes
        # from full plan runs, and the serve smoke's "[v41] RAM plan" line is the
        # end-to-end evidence for them; here the seam pins the freed bytes and their
        # share of a slot (0.45/0.90/1.79 of one expert-per-layer slot at the three ctxs)
        per_cap = int(771.14 * 1024 * 1024)
        for ctx, share in ((65536, 0.44), (131072, 0.89), (262144, 1.78)):
            freed = _state(ctx, True) - _state(ctx, False)
            self.assertEqual(freed, 5510 * ctx, f"freed at ctx {ctx}")
            self.assertGreater(freed / per_cap, share, f"the freed share of a slot at ctx {ctx}")
            self.assertLess(freed / per_cap, share + 0.02, f"the freed share of a slot at ctx {ctx}")

    def test_the_geometry_guard_keeps_the_fp32_price(self):
        # the formats are defined at head_dim 512 / index_head_dim 128: at any other
        # geometry the engine keeps the fp32 host, and the planner must price fp32 there
        # -- the guard declines the packed default exactly as it declines an explicit 0
        self.assertEqual(_state(4096, True, head_dim=64, index_head_dim=32),
                         (2048 + 2048 + 2048 + 4096) * (64 + 32) * 4)
        self.assertEqual(_state(4096, False, head_dim=64, index_head_dim=32),
                         (2048 + 2048 + 2048 + 4096) * (64 + 32) * 4)


if __name__ == "__main__":
    unittest.main()
