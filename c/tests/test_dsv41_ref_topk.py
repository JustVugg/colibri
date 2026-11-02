"""dsv41_ref's top-k selections are the engine's: ties to the lowest index, and the
indexer never keeps a -inf position.

`torch.topk` promises neither. On macOS arm64 it returns [1, 2, 4] for the top 3 of
[0, 0, 0, 0, 1], so the reference generated there picked different indexer blocks
than the one generated on Linux, and `dsv41_tiny` failed 6/8 against an engine that
was right. deepseek_v41.c indexer_run / candidate_blocks take the lowest index with
a strict `>` and leave an empty slot -1; these cases pin the reference to that.
"""
import os
import sys
import unittest

try:
    import torch
except ImportError as e:
    raise unittest.SkipTest(f"torch not installed: {e}")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import dsv41_ref as R  # noqa: E402

INF = float("inf")


class TopkTies(unittest.TestCase):
    def test_ties_go_to_the_lowest_index(self):
        scores = torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0],
                               [2.0, 0.5, 2.0, 0.5, 0.5]])
        self.assertEqual(R.topk_lowest(scores, 3).tolist(), [[4, 0, 1], [0, 2, 1]])

    def test_indexer_ties_at_relu_zero(self):
        # one block wins, four tie at exactly 0.0 for the remaining two slots
        score = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.7]])
        self.assertEqual(R.indexer_pick(score, 3, torch.tensor(5), 10).tolist(), [[10, 11, 14]])

    def test_indexer_leaves_masked_positions_empty(self):
        # a candidate mask left two finite scores for topk=4: the other two slots
        # are empty, not two arbitrary masked blocks
        score = torch.tensor([[-INF, 0.3, -INF, -INF, 0.1, -INF]])
        self.assertEqual(R.indexer_pick(score, 4, torch.tensor(6), 0).tolist(), [[1, 4, -1, -1]])

    def test_indexer_prefill_rows_respect_lens(self):
        # prefill: row t reaches `lens[t]` positions, the rest are -inf
        score = torch.tensor([[0.2, -INF, -INF],
                              [0.2, 0.0, -INF]])
        lens = torch.tensor([[1], [2]])
        self.assertEqual(R.indexer_pick(score, 2, lens, 5).tolist(), [[5, -1], [5, 6]])

    def test_candidate_blocks_tie_to_the_lowest_block(self):
        # four blocks of two; blocks 0, 1 and 2 tie, the last block is pinned in
        logits = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        keep = R.select_candidate_blocks(logits, torch.tensor(8), 2, 2)
        self.assertEqual(keep.int().tolist(), [[1, 1, 0, 0, 0, 0, 1, 1]])


if __name__ == "__main__":
    unittest.main()
