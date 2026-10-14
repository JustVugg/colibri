"""qwen36's Clef path against Clef's own code, on the tiny fixture.

clef_tiny/ref.json holds what the checkpoint's joint_schema_model.py answered
(encode_record, collate_records, ClefModel, in f32, transformers 5.10.2) on the
checkpoint tools/make_clef_tiny.py writes: per case the input ids, the question
and option spans, the option order, the head's logits and the softmax, or the
refusal. tools/make_clef_ref.py wrote it, offline; this test needs neither torch
nor transformers.

Each case goes through the gateway's record builder in the raw form
(openai_server.systemone_decision_record(body, "raw"), what a POST /v1/systemone
sends an engine that asks for it) and the engine's record parser, and must come
back with:

  - the same input ids, exactly (rendering, tokenizer, the cut at max_length);
  - the same spans and the same option order (choice labels sorted, noul true
    first);
  - logits and probabilities within the tolerance ref.json states (f32 weights
    on both sides: the container holds the reference's F16 values exactly; the
    sums run in a different order, and the reference's DeltaNet is chunked where
    the engine's is recurrent);
  - the same decision on every question, and the same refusal.

The engine runs with COLI_DENSE_I8=0 (f32 dense weights) as every qwen36 oracle
does. Skipped when the fixture, its container or the binary is missing
(`make clef-tiny-check` builds all three and sets CLEF_TINY_REQUIRED=1, which
turns the skip into a failure).
"""
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from openai_server import systemone_decision_record  # noqa: E402

FIXTURE = HERE / "clef_tiny"
CONTAINER = HERE / "clef_tiny_c"
BINARY = HERE / ("qwen36.exe" if os.name == "nt" else "qwen36")
REQUIRED = os.environ.get("CLEF_TINY_REQUIRED") == "1"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clef_ready():
    return ((FIXTURE / "ref.json").exists() and (CONTAINER / "joint_head.safetensors").exists()
            and BINARY.exists())


def engine_env(**extra):
    return dict(os.environ, SNAP=str(CONTAINER), COLI_DENSE_I8="0", OMP_NUM_THREADS="2",
                COLI_NO_OMP_TUNE="1", **extra)


def run_engine(**extra):
    result = subprocess.run([str(BINARY), "8", "8"], capture_output=True, text=True,
                            encoding="utf-8", env=engine_env(**extra), timeout=600)
    if result.returncode:
        raise AssertionError(result.stderr[-3000:])
    return result.stdout


class ClefTinyOracle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not clef_ready():
            message = "clef_tiny, clef_tiny_c or qwen36 missing: run make clef-tiny-check"
            if REQUIRED:
                raise AssertionError(message)
            raise unittest.SkipTest(message)
        cls.ref = json.loads((FIXTURE / "ref.json").read_text(encoding="utf-8"))
        for name, digest in cls.ref["fixture_sha256"].items():
            if sha256(FIXTURE / name) != digest:
                raise AssertionError(f"clef_tiny/{name} is not the file ref.json was computed on; "
                                     "regenerate it with make clef-tiny-generate (or the reference "
                                     "with tools/make_clef_ref.py if the generator changed)")

    def test_tokenizer_matches_the_reference(self):
        strings = [entry["text"] for entry in self.ref["tokenizer"]]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "strings.json"
            path.write_text(json.dumps(strings, ensure_ascii=False), encoding="utf-8")
            got = json.loads(run_engine(CLEF_TOKENIZE=str(path)))
        self.assertEqual(len(got), len(strings))
        for entry, ids in zip(self.ref["tokenizer"], got):
            self.assertEqual(ids, entry["ids"], f"tokenizer: {entry['text']!r}")

    def run_cases(self, cases, max_length=None):
        records = [{"payload": json.dumps(systemone_decision_record(
            {"state": case["state"], "questions": case["questions"]}, "raw"), ensure_ascii=False)}
            for case in cases]
        extra = {"CLEF_RECORDS": "", "CLEF_IDS": "1"}
        if max_length:
            extra["COLI_CLEF_MAX_LEN"] = str(max_length)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "records.json"
            path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
            extra["CLEF_RECORDS"] = str(path)
            lines = run_engine(**extra).splitlines()
        self.assertEqual(len(lines), len(cases))
        return [json.loads(line) for line in lines]

    def test_every_case_matches_the_reference(self):
        tol = self.ref["tolerance"]
        groups = {}
        for case in self.ref["cases"]:
            groups.setdefault(case.get("max_length"), []).append(case)
        worst = {"probs": 0.0, "logits": 0.0}
        decisions = 0
        for max_length, cases in groups.items():
            for case, got in zip(cases, self.run_cases(cases, max_length)):
                name = case["name"]
                if "error" in case:
                    self.assertIn("error", got, f"{name}: the reference refused, the engine answered")
                    # "schema requires N tokens before state; maximum is M", in the
                    # engine's words and naming the field: the same two numbers
                    self.assertTrue(got["error"].startswith("questions: "), got["error"])
                    self.assertEqual(re.findall(r"\d+", got["error"]), re.findall(r"\d+", case["error"]), name)
                    continue
                self.assertNotIn("error", got, f"{name}: {got.get('error')}")
                self.assertEqual(got["input_ids"], case["input_ids"], f"{name}: input ids")
                self.assertEqual(got["input_tokens"], len(case["input_ids"]), name)
                record = systemone_decision_record({"state": case["state"],
                                                    "questions": case["questions"]}, "raw")
                self.assertEqual(len(got["answers"]), len(case["answers"]), name)
                for want, answer, span, question in zip(case["answers"], got["answers"], got["spans"],
                                                        record["questions"]):
                    where = f"{name}.{want['id']}"
                    self.assertEqual(answer["id"], want["id"], where)
                    self.assertEqual(span["type"], want["type"], where)
                    self.assertEqual(span["question"], want["question_span"], f"{where}: question span")
                    self.assertEqual(span["options"], want["option_spans"], f"{where}: option spans")
                    labels = [option["label"] for option in question["options"]]
                    order = [labels.index(option) for option in want["option_ids"]]
                    self.assertEqual(span["order"], order, f"{where}: option order")
                    for key in ("logits", "probs"):
                        diff = max(abs(answer[key][i] - want[key][k]) for k, i in enumerate(order))
                        worst[key] = max(worst[key], diff)
                        self.assertLessEqual(diff, tol[key], f"{where}: {key} {answer[key]} vs {want[key]}")
                    pick = max(range(len(order)), key=lambda k: want["probs"][k])
                    mine = max(range(len(answer["probs"])), key=lambda i: answer["probs"][i])
                    self.assertEqual(mine, order[pick], f"{where}: a different decision")
                    decisions += 1
        sys.stderr.write(f"\n[clef tiny] {decisions} decisions identical; max |dp| {worst['probs']:.2e}, "
                         f"max |dlogit| {worst['logits']:.2e}\n")


if __name__ == "__main__":
    unittest.main()
