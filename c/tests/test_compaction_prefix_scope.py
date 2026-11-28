"""A cached compaction summary may contain only the prefix its key identifies."""
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import openai_server as gateway


def turn(role, text):
    return {'role': role, 'content': text}


class RecordingCompactor(gateway.APIHandler):
    """Keep summary input visible rather than relying on a model to disclose it."""
    def __init__(self):
        self.server = SimpleNamespace(context_limit=1000, tokens_per_char=0.25,
                                      kv_slots=1, compactions=gateway.CompactionMemory())
        self.covered = []

    def _summarize(self, head, part, render, words, budget, cache_slot, limit, ratio):
        self.covered.extend(part)
        return '|'.join(m['content'] for m in part)


class CompactionPrefixScopeTest(unittest.TestCase):
    def setUp(self):
        self.handler = RecordingCompactor()
        self.lead = [turn('system', 'Shared policy')]
        self.conversation = [turn('user', 'old ' * 400),
                             turn('assistant', 'old reply ' * 100),
                             turn('user', 'Branch A private choice'),
                             turn('assistant', 'Branch A acknowledgement'),
                             turn('user', 'Latest question')]
        self.keys = gateway.compaction_keys(self.lead, self.conversation)

    def compact(self):
        return self.handler._compact({}, self.lead + self.conversation, self.lead,
                                     self.conversation, self.keys, 0, None, {})

    def test_summary_covers_exactly_the_prefix_replaced(self):
        done, summary = self.compact()
        self.assertEqual(done, 2)  # the final three short messages stay verbatim
        self.assertEqual(self.handler.covered, self.conversation[:done])
        self.assertNotIn('Branch A private choice', summary)
        self.assertEqual(self.handler.server.compactions.find(self.keys), (done, summary))

    def test_fork_cannot_reuse_facts_after_its_shared_prefix(self):
        done, summary = self.compact()
        fork = self.conversation[:2] + [turn('user', 'Branch B changed choice'),
                                       turn('assistant', 'Branch B acknowledgement'),
                                       turn('user', 'Different question')]
        fork_keys = gateway.compaction_keys(self.lead, fork)
        cached_done, cached = self.handler.server.compactions.find(fork_keys)
        self.assertEqual(cached_done, done)
        self.assertEqual(cached, summary)
        # A summary at this key is valid in both branches only if no branch-specific
        # messages were supplied to the summarizer in the first place.
        self.assertTrue(all(message in fork[:cached_done] for message in self.handler.covered))
        self.assertNotIn('Branch A', cached)


if __name__ == '__main__':
    unittest.main()
