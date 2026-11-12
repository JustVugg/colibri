"""The CUDA chain's host-KV pointer arithmetic must be BYTE arithmetic.

`Layer.window/ckv/ikey` are `uint8_t *` since the packed-KV rework, so
`l->ckv + (size_t)t0 * head_dim` is a byte offset of t0*512 into rows that are
hd*4 = 2048 B wide even in the fp32 opt-out (and 288 B under the packed
default). As a `float *` the same expression was an ELEMENT offset = t0*2048 B,
so the re-type silently quartered every CUDA-chain row stride in both modes:
the pull's 16K write-back runs 14,417,920 B past a 2,359,296 B compressed-row
buffer under the default, and at COLI_VK_KV_FP32=1 the rows overlap 4:1.

Every such offset therefore goes through the byte-stride helpers
(`v41_row_win/ckv/ikey`, deepseek_v41.c) -- and the chain declines a packed host
outright in `v41u_wanted`, so a CUDA build never reads code rows as floats. What the
tier LEAVES the chain (`VktConfig.chain_bytes`, `v41_tier_chain_bytes`) has to follow
that same decline, or the device holds expert slots for a chain that never runs.
Static: no CUDA device is needed, and none is used (the runtime gate would be
tests/cuda_tier_engines.sh, which wants a card).
"""
import re
import unittest
from pathlib import Path

CUCHAIN = Path(__file__).resolve().parent.parent / "deepseek_v41_cuchain.h"
ENGINE = Path(__file__).resolve().parent.parent / "deepseek_v41.c"

ELEMENT_STRIDE = re.compile(r"l->(?:window|ckv|ikey)\b\s*\+\s*[^,;{}]*?\*\s*(?:hd|ihd)\b")
HELPERS = ("v41_row_win", "v41_row_ckv", "v41_row_ikey")


def cuchain_text():
    return CUCHAIN.read_text()


def engine_text():
    return ENGINE.read_text()


def wanted_body(text):
    start = text.index("static int v41u_wanted(Model *m) {")
    end = text.index("\n}\n", start)
    return text[start:end]


def function_body(text, signature):
    start = text.index(signature)
    end = text.index("\n}\n", start)
    return text[start:end]


class HostKvStrideTest(unittest.TestCase):
    def test_no_element_stride_on_a_ret_typed_host_row(self):
        bad = [m.group(0) for m in ELEMENT_STRIDE.finditer(cuchain_text())]
        self.assertEqual([], bad, "fp32 row stride on a uint8_t* host pointer (byte-scaled, "
                                 "1/4 of a row):\n  " + "\n  ".join(bad))

    def test_offsets_go_through_the_byte_stride_helpers(self):
        text = cuchain_text()
        missing = [h for h in HELPERS if h not in text]
        self.assertEqual([], missing, "no byte-stride helper in the file: " + ", ".join(missing))

    def test_the_chain_declines_a_packed_host(self):
        body = wanted_body(cuchain_text())
        self.assertIn("v41_host_kvq(&m->c)", body,
                      "v41u_wanted accepts a build whose host rows are codes: the CUDA chain "
                      "would read 528/288/68 B rows as fp32")


class TierChainReservationTest(unittest.TestCase):
    """The other side of the same decline: what the tier reserves for the chain.

    `v41_tier_chain_bytes` is `VktConfig.chain_bytes` (deepseek_v41.c:6782), subtracted
    from the device's expert room (vk_tier.c: "what the dense chain allocates at its
    first forward"). It gated on COLI_CUDA_CHAIN alone, so the packed default -- where
    `v41u_wanted` has already declined the chain -- still held the chain's own
    v41u_plan_bytes on the device: E*LR*4 for the candidate lists plus the P*E mask,
    >= 1.74 GB at 262144, lost to the expert slots on every conversation.
    """

    def test_the_tier_reserves_nothing_for_the_chain_a_packed_host_declines(self):
        body = function_body(engine_text(), "static size_t v41_tier_chain_bytes(Model *m) {")
        self.assertIn("!v41_host_kvq(&m->c)", body,
                      "v41_tier_chain_bytes holds device memory for the chain v41u_wanted "
                      "declines on a packed host (the default): the host condition has to be "
                      "in the reservation's guard too")

    def test_the_reservation_still_prices_an_fp32_host(self):
        # the floor of the fix: an fp32 host (COLI_VK_KV_FP32=1) runs the chain, so its
        # prompt has to keep the device room -- the whole point of chain_bytes
        body = function_body(engine_text(), "static size_t v41_tier_chain_bytes(Model *m) {")
        for needed in ("xt_cuda()", "m->gguf", "g_v41_mux_slots",
                       "v41u_plan_bytes(m, m->c.max_positions)"):
            self.assertIn(needed, body, f"v41_tier_chain_bytes no longer gates on {needed}")


if __name__ == "__main__":
    unittest.main()
