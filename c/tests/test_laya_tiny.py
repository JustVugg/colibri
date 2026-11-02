"""laya.c against the `laya` package, on the tiny fixture.

laya_tiny/ref.json holds what Convai Innovations' own runtime (laya.Agent,
pinned version) answered on the checkpoint tools/make_laya_tiny.py writes:
per question the token ids, the [MASK] marker positions, the raw scorer
logits, the calibrated probabilities and the act head, plus the refusal for a
question whose options do not fit. tools/make_laya_ref.py wrote it, offline;
this test needs neither torch nor the package.

Each case goes through the gateway's own record builder
(openai_server.systemone_decision_record) and the engine's record parser, the
path a POST /v1/systemone takes, and must come back with:

  - the same token ids and markers, exactly (tokenizer, sequence assembly,
    truncation of long states and conversations, the option budget);
  - logits within 2e-4 and probabilities within 2e-5 of the reference, the
    tolerance ref.json states (f32 against f32, different summation order);
  - the same decision (argmax) on every question;
  - the same refusal text when the reference refused.

The tokenizer is also checked alone on strings chosen to hit NFC, added tokens
matched before BPE (runs of spaces, the |||...||| placeholders, specials),
contractions, digits, non-Latin scripts and emoji.

Skipped when the fixture or the binary is missing (`make laya-tiny-check`
builds both and sets LAYA_TINY_REQUIRED=1, which turns the skip into a failure).
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

FIXTURE = HERE / "laya_tiny"
BINARY = HERE / ("laya.exe" if os.name == "nt" else "laya")
REQUIRED = os.environ.get("LAYA_TINY_REQUIRED") == "1"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class LayaTinyOracle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ref = json.loads((FIXTURE / "ref.json").read_text(encoding="utf-8"))
        missing = [str(p) for p in (FIXTURE / "model.safetensors", BINARY) if not p.exists()]
        if missing:
            message = f"missing {', '.join(missing)}: run make laya-tiny-check"
            if REQUIRED:
                raise AssertionError(message)
            raise unittest.SkipTest(message)
        for name, digest in cls.ref["fixture_sha256"].items():
            if sha256(FIXTURE / name) != digest:
                raise AssertionError(f"laya_tiny/{name} is not the file ref.json was computed on; "
                                     "regenerate it with make laya-tiny-generate (or the reference "
                                     "with tools/make_laya_ref.py if the generator changed)")

    def run_engine(self, *args):
        result = subprocess.run([str(BINARY), "--model", str(FIXTURE), *args],
                                capture_output=True, text=True, encoding="utf-8",
                                env=dict(os.environ, OMP_NUM_THREADS="2"), timeout=300)
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        return result.stdout

    def test_tokenizer_matches_the_reference(self):
        strings = [entry["text"] for entry in self.ref["tokenizer"]]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "strings.json"
            path.write_text(json.dumps(strings, ensure_ascii=False), encoding="utf-8")
            got = json.loads(self.run_engine("--tokenize", str(path)))
        for entry, ids in zip(self.ref["tokenizer"], got):
            self.assertEqual(ids, entry["ids"], f"tokenizer: {entry['text']!r}")
        self.assertEqual(len(got), len(strings))

    def test_every_case_matches_the_reference(self):
        cases = self.ref["cases"]
        tol = self.ref["tolerance"]
        records = []
        for case in cases:
            record = systemone_decision_record({"state": case["state"],
                                                "questions": case["questions"]})
            records.append({"payload": json.dumps(record, ensure_ascii=False)})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "records.json"
            path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
            lines = self.run_engine("--records", str(path), "--ids").splitlines()
        self.assertEqual(len(lines), len(cases))
        worst = {"probs": 0.0, "logits": 0.0, "act_probability": 0.0}
        decisions = 0
        for case, line in zip(cases, lines):
            got = json.loads(line)
            name = case["name"]
            if "error" in case:
                self.assertIn("error", got, f"{name}: the reference refused, the engine answered")
                # the engine names the field; the reference names the question
                want = case["error"].split(": ", 1)[1]
                self.assertTrue(got["error"].endswith(want), f"{name}: {got['error']!r} vs {want!r}")
                continue
            self.assertNotIn("error", got, f"{name}: {got.get('error')}")
            self.assertEqual(len(got["answers"]), len(case["answers"]), name)
            for want, answer, seq in zip(case["answers"], got["answers"], got["sequences"]):
                where = f"{name}.{want['id']}"
                self.assertEqual(answer["id"], want["id"], where)
                self.assertEqual(seq["ids"], want["ids"], f"{where}: token ids")
                self.assertEqual(seq["markers"], want["markers"], f"{where}: markers")
                self.assertEqual(answer["tokens"], len(want["ids"]), where)
                self.assertEqual(answer["state_tokens"], want["state_tokens"], where)
                self.assertEqual(answer["state_dropped"], want["state_dropped"], where)
                self.assertAlmostEqual(answer["temperature"], want["temperature"], places=6, msg=where)
                for key in ("logits", "probs"):
                    diff = max(abs(a - b) for a, b in zip(answer[key], want[key]))
                    worst[key] = max(worst[key], diff)
                    self.assertLessEqual(diff, tol[key], f"{where}: {key} {answer[key]} vs {want[key]}")
                act = answer["actions"]["act"]
                worst["act_probability"] = max(worst["act_probability"], abs(act - want["act_probability"]))
                self.assertLessEqual(abs(act - want["act_probability"]), tol["act_probability"], where)
                pick = max(range(len(answer["probs"])), key=lambda i: (answer["probs"][i], -i))
                ref_pick = max(range(len(want["probs"])), key=lambda i: (want["probs"][i], -i))
                self.assertEqual(pick, ref_pick, f"{where}: a different decision")
                decisions += 1
        sys.stderr.write(f"\n[laya tiny] {decisions} decisions identical; max |dp| {worst['probs']:.2e}, "
                         f"max |dlogit| {worst['logits']:.2e}, max |dact| {worst['act_probability']:.2e}\n")


if __name__ == "__main__":
    unittest.main()
