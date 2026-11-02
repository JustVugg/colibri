"""The stream ring's stage pool must outlive its first job (PW-B2).

v4_vkt_ring_batch (COLI_VK_TIER_STREAM_RING's reader) hands each expert it reads a
buffer of a stage pool of the ring's own -- a second pool beside the engine thread's
bypass pool, so the two never share a busy flag -- and the tier's release hook
(v4_vkt_unhold) hands those buffers back. PW-B's first cut keyed that hand-back on the
hold's stage index alone, which indexes whichever pool handed it out: a ring hold's
release freed the ENGINE pool's flag and left the ring's busy forever. The ring then
filled its first job's worth (the pool's whole 8-buffer capacity) and nothing after --
W2's `the ring 42 jobs (8 experts)` line was this leak compounding the empty plans --
and each such release also falsely freed an engine stage a concurrent engine fill could
be reading into.

Silent on output (the inline fill covers whatever the ring does not fill) and visible
only as fill counts at model scale, which no fixture without a 43-layer model reaches:
like test_v4_moe_batch_lifecycle_source, this pins the shape at source level. The
runtime proof is the engagement line -- with COLI_VK_TIER_STREAM_RING_COLD every cold
job fills its whole plan (deepseek-v4-smoke with the ring and the cold rule on).
"""
import unittest
from pathlib import Path

SOURCE = Path(__file__).resolve().parent.parent / "deepseek_v4.c"


def find(text, needle, start=0, where="deepseek_v4.c"):
    """text.index(needle, start), except that a missing needle fails the test
    with a message naming it rather than erroring with a bare ValueError."""
    at = text.find(needle, start)
    if at < 0:
        raise AssertionError(f"{where}: no {needle!r}")
    return at


def function_body(text, signature):
    start = find(text, signature)
    end = find(text, "\n}\n", start)
    return text[start:end]


class V4VkRingPoolSourceTest(unittest.TestCase):
    def setUp(self):
        self.text = SOURCE.read_text(encoding="utf-8")

    def test_the_hold_carries_its_pool(self):
        """A hold names the pool its stage indexes, or the release cannot tell."""
        body = function_body(self.text, "typedef struct { ColiExpertView view")
        self.assertIn("int stage;", body)
        self.assertIn("int ring;", body)

    def test_the_release_frees_the_pool_the_hold_came_from(self):
        """Both directions: a ring hold frees the ring's flag, an engine hold the
        engine's -- and a lease hold (stage < 0) releases the store's view."""
        body = function_body(self.text, "static void v4_vkt_unhold(void *ctx, void *h)")
        branch = find(body, "if (hold->ring)", where="v4_vkt_unhold")
        ring = find(body, "v4_vkt_ring_stage_back(hold->stage)", branch, where="v4_vkt_unhold, the ring branch")
        engine = find(body, "else if (hold->stage >= 0)", ring, where="v4_vkt_unhold, the engine branch")
        find(body, "v4_vkt_stage_busy[hold->stage] = 0", engine,
             where="v4_vkt_unhold, the engine branch frees the engine pool")
        find(body, "coli_expert_release((ColiExpertStore *)ctx, &hold->view)", engine,
             where="v4_vkt_unhold, the lease branch")
        self.assertNotIn("v4_vkt_ring_stage_busy[hold->stage]", body,
                         "v4_vkt_unhold frees the ring pool by index again: the ring and "
                         "engine pools must go through their own hand-back, never a shared array")

    def test_the_ring_reader_marks_its_holds(self):
        """Every hold the ring's reader returns is a ring hold; the engine's reader
        (v4_vkt_load_one) must not mark one, or an engine stage would free the ring's."""
        ring = function_body(self.text, "static int v4_vkt_ring_batch(void *ctx")
        marked = find(ring, "hold->ring = 1;", where="v4_vkt_ring_batch")
        find(ring, "hold->stage = st[i];", 0, where="v4_vkt_ring_batch")
        self.assertLessEqual(find(ring, "hold->stage = st[i];"), marked)
        engine = function_body(self.text, "static int v4_vkt_load_one(ColiExpertStore *store")
        self.assertNotIn("hold->ring = 1", engine, "v4_vkt_load_one must leave the ring mark unset")
        self.assertIn("hold->stage = -1;", engine)

    def test_no_other_hand_back(self):
        """The ring pool's busy flag changes in exactly three places: the reader's own
        failure paths, its hand-back helper -- and nothing else (an engine-path write
        would be the leak again). Matched by identifier, not by spelling: the
        declaration's qualifiers and the take/write sites may be reflowed, and the
        pool's own lifecycle sites (STREAM_DEPTH's calloc, its shutdown reset, the
        free and the allocation guard) read and rebind the pointer without a flag
        ever changing."""
        writes = [i for i in range(len(self.text)) if self.text.startswith("v4_vkt_ring_stage_busy", i)]
        self.assertTrue(writes)
        for at in writes:
            line = self.text[at:self.text.find("\n", at)]
            self.assertTrue(
                ";" in line.strip() and "=" not in line          # the declaration, whatever its qualifiers
                or "v4_vkt_ring_stage_busy[stage] = 0" in line    # the hand-back helper
                or "v4_vkt_ring_stage_busy[st[i]] = 0" in line    # the reader's own failure paths
                or "v4_vkt_ring_stage_busy[j] = 1" in line        # the reader's take
                or line.startswith(("v4_vkt_ring_stage_busy = calloc",   # the pool's creation (STREAM_DEPTH)
                                    "v4_vkt_ring_stage_busy = NULL",     # its shutdown reset
                                    "v4_vkt_ring_stage_busy)")),         # free() and the allocation guard: the pointer, not a flag
                f"an unexpected hand on the ring's stage pool: {line.strip()!r}",
            )


if __name__ == "__main__":
    unittest.main()
