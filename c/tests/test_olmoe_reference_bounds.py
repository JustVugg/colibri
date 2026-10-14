"""OLMoE checks the reference's lengths and ids before the run.

The one-shot mode reads prompt_ids and full_ids from a JSON reference and
indexes with them: full_ids' length sizes the KV cache, every position up to it
writes one score into attention()'s per-head buffer, and every id selects an
embedding row (and, under PPL=1, a logit). Nothing checked them:

  * a reference longer than 4096 tokens ran past the score buffer, a stack
    array of 4096 floats (CTX caps chat and serve at 4096 for that reason; the
    one-shot and PPL=1 paths had no cap);
  * full_ids no longer than prompt_ids, or an empty prompt, sized the token
    buffer and the KV cache shorter than the prefill writes;
  * an id outside [0, vocab) read past the embedding table.

What is asserted is what the real engine does with each reference on a real
container: a refusal that names the problem, before any token is scored. Needs
a converted tiny OLMoE container and a built engine, the ones the OLMoE tiny
oracle job builds.
"""
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
ENGINE = HERE / ("olmoe.exe" if os.name == "nt" else "olmoe")
FIXTURE = Path(os.environ.get("COLI_OLMOE_FIXTURE", ""))
MAX_POSITIONS = 4096
VOCAB = 128                      # the tiny container's


def tokens(n, start=0):
    """n token ids inside the tiny vocabulary."""
    return [(start + i) % 100 + 1 for i in range(n)]


def run(prompt_ids, full_ids, **env):
    with tempfile.TemporaryDirectory() as tmp:
        ref = Path(tmp) / "ref.json"
        ref.write_text(json.dumps({"prompt_ids": prompt_ids, "full_ids": full_ids}))
        environment = {**os.environ, "SNAP": str(FIXTURE), **env}
        return subprocess.run([str(ENGINE), "8", "8", str(ref)], env=environment,
                              capture_output=True, text=True, errors="replace",
                              timeout=600)


@unittest.skipUnless(ENGINE.exists(), "olmoe engine is not built")
@unittest.skipUnless(FIXTURE.name and (FIXTURE / "config.json").is_file(),
                     "COLI_OLMOE_FIXTURE not set to a converted OLMoE container")
class OlmoeReferenceBoundsTest(unittest.TestCase):

    def assertRefused(self, result, reason):
        self.assertEqual(result.returncode, 1, result.stderr[-2000:])
        self.assertIn(reason, result.stderr, result.stderr[-2000:])
        self.assertNotIn("Matching tokens", result.stdout)
        self.assertNotIn("TF-NLL", result.stdout)

    def test_a_reference_that_fills_the_context_runs(self):
        full = tokens(MAX_POSITIONS)
        result = run(full[:MAX_POSITIONS - 8], full)
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        self.assertIn("Matching tokens: ", result.stdout)
        self.assertIn("/8\n", result.stdout)

    def test_one_token_more_than_the_context_is_refused(self):
        full = tokens(MAX_POSITIONS + 1)
        self.assertRefused(run(full[:MAX_POSITIONS - 8], full),
                           f"more than the {MAX_POSITIONS} positions")

    def test_a_reference_past_the_score_buffer_is_refused(self):
        full = tokens(MAX_POSITIONS + 200)
        self.assertRefused(run(full[:MAX_POSITIONS], full),
                           f"more than the {MAX_POSITIONS} positions")

    def test_ppl_refuses_it_too(self):
        full = tokens(MAX_POSITIONS + 200)
        self.assertRefused(run(full[:16], full, PPL="1"),
                           f"more than the {MAX_POSITIONS} positions")

    def test_full_ids_shorter_than_the_prompt_is_refused(self):
        full = tokens(64)
        self.assertRefused(run(full, full[:8]), "need 1 <= prompt < full")

    def test_ppl_refuses_full_ids_shorter_than_the_prompt(self):
        full = tokens(64)
        self.assertRefused(run(full, full[:8], PPL="1"), "need 1 <= prompt < full")

    def test_nothing_after_the_prompt_is_refused(self):
        full = tokens(16)
        self.assertRefused(run(full, full, PPL="1"), "need 1 <= prompt < full")

    def test_an_empty_prompt_is_refused(self):
        self.assertRefused(run([], tokens(8)), "need 1 <= prompt < full")

    def test_a_prompt_id_past_the_vocabulary_is_refused(self):
        full = tokens(16)
        prompt = full[:8]
        prompt[3] = VOCAB
        self.assertRefused(run(prompt, full), f"prompt_ids[3] = {VOCAB} is not a token id")

    def test_a_negative_full_id_is_refused(self):
        full = tokens(16)
        full[15] = -1                # the last one: under PPL=1 it indexes the logits
        self.assertRefused(run(full[:8], full, PPL="1"), "full_ids[15] = -1 is not a token id")


if __name__ == "__main__":
    unittest.main()
