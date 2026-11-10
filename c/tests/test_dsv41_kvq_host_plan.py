"""The packed cache's planner re-price (family_registry._dsv41_geometry): the host's
KV state priced as the cache's packed code rows while COLI_VK_KV_FP32 is unset (the
default now -- unset runs packed, mirrors and host alike), fp32 under =1 (the exact
opt-out, which takes the device mirrors with it).

What is pinned here is the SEAM -- the per-row width swap and its guard, at the released
geometry's config and the three context lengths whose slot deltas the memory model
predicted (+1/+1/+2 at 64K/128K/256K against per_cap 771.14 MiB). The slot arithmetic
itself (cache_bytes // per_cap) needs the checkpoint's own tensors for per_cap, so its
end-to-end evidence is the serve smoke's "[v41] RAM plan" line, the packed default
against FP32=1 -- the same line the ladder reproduced for every measured context.
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
# window 128, head_dim 512, index_head_dim 128, 3 DSpark stages.
def _config(**over):
    ratios = [0] * 40
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
        "n_mtp_layers": 3,
        "moe_inter_dim": 2048,
    }
    cfg.update(over)
    return {"text_config": cfg}


def _state(context, fp32, **over):
    """The context state bytes at a flag setting. fp32 True: COLI_VK_KV_FP32=1 (the
    exact opt-out); False: the var UNSET -- the default, packed."""
    env = dict(os.environ)
    env.pop("COLI_VK_KV_FP32", None)
    if fp32:
        env["COLI_VK_KV_FP32"] = "1"
    old = os.environ
    try:
        os.environ = env
        return fr._dsv41_geometry(_config(**over), context, None).context_state_bytes
    finally:
        os.environ = old


class Dsv41KvqHostPlanTest(unittest.TestCase):
    def test_the_packed_price_the_fp32_one_and_the_default(self):
        # a position costs 2.5 compressed rows: 6400 B fp32, 890 B packed -- and the
        # var unset IS the packed price (the engine's default, mirrored here)
        for ctx in (65536, 131072, 262144):
            self.assertEqual(_state(ctx, True), 6400 * ctx, f"fp32 state at ctx {ctx}")
            self.assertEqual(_state(ctx, False), 890 * ctx, f"packed state at ctx {ctx}")

    def test_the_freed_ram_against_the_slot_it_buys(self):
        # 5510 B a position freed. The slot a freed budget BUYS is the floor of the whole
        # cache_bytes / per_cap, not the floor of the freed bytes alone: the ladder's
        # +1/+1/+2 at 64K/128K/256K (per_cap 771.14 MiB, one expert of every layer) comes
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
