"""gliner_decide.c against the `gliner2` package, on the tiny fixture.

gliner_decide_tiny/ref.json holds what fastino's own runtime (gliner2's
AutoExtractor.classify_text, pinned version) answered on the checkpoint
tools/make_gliner_decide_tiny.py writes: per case the encoder's input ids, the
[L] marker positions, the classifier's logits and the softmax over each task's
labels, plus the refusal the engine owes when the questions alone exceed
max_len. tools/make_gliner_decide_ref.py wrote it, offline; this test needs
neither torch nor the package.

Each case goes through the gateway's own record builder
(openai_server.systemone_decision_record) and the engine's record parser, the
path a POST /v1/systemone takes, with COLI_GLINER_MAX_LEN at the value the
reference cut the long states with, and must come back with:

  - the same token ids and markers, exactly (word splitter, lower-casing,
    tokenizer, the schema of every question, the state cut at whole words);
  - logits within 1e-4 and probabilities within 2e-5 of the reference, the
    tolerance ref.json states (f32 against f32, different summation order);
  - the same decision (argmax) on every question;
  - the refusal, naming the questions, when they do not fit.

The tokenizer and the word splitter are also checked alone, on strings chosen
to hit the normalizer (whitespace runs, NFC), the added tokens, unknown
characters, URLs, e-mails, @handles, final sigma, dotted capital I and emoji.

Skipped when the fixture or the binary is missing (`make gliner-decide-tiny-check`
builds both and sets GLINER_DECIDE_TINY_REQUIRED=1, which turns the skip into a
failure).
"""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from openai_server import systemone_decision_record  # noqa: E402

FIXTURE = HERE / "gliner_decide_tiny"
BINARY = HERE / ("gliner_decide.exe" if os.name == "nt" else "gliner_decide")
REQUIRED = os.environ.get("GLINER_DECIDE_TINY_REQUIRED") == "1"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class GlinerDecideTinyOracle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ref = json.loads((FIXTURE / "ref.json").read_text(encoding="utf-8"))
        missing = [str(p) for p in (FIXTURE / "model.safetensors", BINARY) if not p.exists()]
        if missing:
            message = f"missing {', '.join(missing)}: run make gliner-decide-tiny-check"
            if REQUIRED:
                raise AssertionError(message)
            raise unittest.SkipTest(message)
        for name, digest in cls.ref["fixture_sha256"].items():
            if sha256(FIXTURE / name) != digest:
                raise AssertionError(f"gliner_decide_tiny/{name} is not the file ref.json was computed "
                                     "on; regenerate it with make gliner-decide-tiny-generate (or the "
                                     "reference with tools/make_gliner_decide_ref.py if the generator "
                                     "changed)")

    def run_engine(self, *args, model=True):
        argv = [str(BINARY), *(("--model", str(FIXTURE)) if model else ()), *args]
        env = dict(os.environ, OMP_NUM_THREADS="2", COLI_GLINER_MAX_LEN=str(self.ref["max_len"]))
        result = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8", env=env,
                                timeout=300)
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        return result.stdout

    def run_strings(self, flag, strings, model=True):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "strings.json"
            path.write_text(json.dumps(strings, ensure_ascii=False), encoding="utf-8")
            return json.loads(self.run_engine(flag, str(path), model=model))

    def test_tokenizer_matches_the_reference(self):
        got = self.run_strings("--tokenize", [entry["text"] for entry in self.ref["tokenizer"]])
        self.assertEqual(len(got), len(self.ref["tokenizer"]))
        for entry, ids in zip(self.ref["tokenizer"], got):
            self.assertEqual(ids, entry["ids"], f"tokenizer: {entry['text']!r}")

    def test_word_splitter_matches_the_reference(self):
        got = self.run_strings("--split", [entry["text"] for entry in self.ref["split"]], model=False)
        self.assertEqual(len(got), len(self.ref["split"]))
        for entry, words in zip(self.ref["split"], got):
            self.assertEqual(words, entry["words"], f"split: {entry['text']!r}")

    def test_every_case_matches_the_reference(self):
        cases = self.ref["cases"]
        tol = self.ref["tolerance"]
        records = []
        for case in cases:
            record = systemone_decision_record({"state": case["state"], "questions": case["questions"]})
            records.append({"payload": json.dumps(record, ensure_ascii=False)})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "records.json"
            path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
            lines = self.run_engine("--records", str(path), "--ids").splitlines()
        self.assertEqual(len(lines), len(cases))
        worst = {"probs": 0.0, "logits": 0.0}
        decisions = cut = 0
        for case, line in zip(cases, lines):
            got = json.loads(line)
            name = case["name"]
            if "refused" in case:
                self.assertIn("error", got, f"{name}: the questions do not fit, the engine answered")
                self.assertTrue(got["error"].startswith(case["refused"]), f"{name}: {got['error']!r}")
                continue
            self.assertNotIn("error", got, f"{name}: {got.get('error')}")
            seq = got["sequence"]
            self.assertEqual(seq["ids"], case["ids"], f"{name}: token ids")
            self.assertEqual(seq["markers"], case["markers"], f"{name}: [L] markers")
            self.assertEqual((seq["words"], seq["words_used"]), (case["words"], case["words_used"]), name)
            self.assertEqual(got["input_tokens"], len(case["ids"]), name)
            cut += case["words_used"] < case["words"]
            self.assertEqual(len(got["answers"]), len(case["answers"]), name)
            for want, answer in zip(case["answers"], got["answers"]):
                where = f"{name}.{want['id']}"
                self.assertEqual(answer["id"], want["id"], where)
                self.assertEqual(answer["tokens"], len(case["ids"]), where)
                self.assertEqual(answer["state_tokens"], case["state_tokens"], where)
                self.assertEqual(answer["state_dropped"], case["state_tokens"] - case["state_used"], where)
                self.assertIsNone(answer["temperature"], where)
                for key in ("logits", "probs"):
                    self.assertEqual(len(answer[key]), len(want[key]), where)
                    diff = max(abs(a - b) for a, b in zip(answer[key], want[key]))
                    worst[key] = max(worst[key], diff)
                    self.assertLessEqual(diff, tol[key], f"{where}: {key} {answer[key]} vs {want[key]}")
                pick = max(range(len(answer["probs"])), key=lambda i: (answer["probs"][i], -i))
                ref_pick = max(range(len(want["probs"])), key=lambda i: (want["probs"][i], -i))
                self.assertEqual(pick, ref_pick, f"{where}: a different decision")
                decisions += 1
        self.assertGreaterEqual(cut, 2, "the fixture must cut at least two long states")
        sys.stderr.write(f"\n[gliner_decide tiny] {decisions} decisions identical; max |dp| "
                         f"{worst['probs']:.2e}, max |dlogit| {worst['logits']:.2e}\n")


if __name__ == "__main__":
    unittest.main()
