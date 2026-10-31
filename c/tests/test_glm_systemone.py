"""/v1/systemone on colibri.c (the GLM family: GLM-5.2, Mistral-Large-4), end to end.

colibri.c reads seven numeric fields in a SUBMIT header (the grammar's byte count is the
seventh) before its key=value tokens. The chat sent it; System One's scoring did not, so
colibri.c refused the header, answered with request id 0, and the request waited for an
answer that never came: the System One page on Mistral-Large-4 never answered.

Through the gateway and the real engine on the tiny GLM oracle (tools/make_glm_oracle.py,
the byte tokenizer test_glm_prompt_cancel.py writes into a scratch copy):
  - a choice question and a yes/no question are answered, the probabilities summing to 1;
  - a header the engine cannot read is answered by its own id (the request ends with an
    error instead of waiting), and its payload is not read as commands.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import openai_server  # noqa: E402
from family_registry import family_by_id  # noqa: E402
from openai_server import APIServer  # noqa: E402
from test_glm_prompt_cancel import BINARY, FIXTURE, byte_tokenizer  # noqa: E402


@unittest.skipUnless(BINARY, "colibri is not built")
@unittest.skipUnless(FIXTURE.name and (FIXTURE / "model.safetensors").is_file(),
                     "COLI_ORACLE_FIXTURE not set to the tiny GLM oracle (tools/make_glm_oracle.py)")
class GlmSystemOne(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="glm_s1_")
        cls.snap = Path(cls.tmp) / "snap"
        shutil.copytree(FIXTURE, cls.snap)
        (cls.snap / "tokenizer.json").write_text(json.dumps(byte_tokenizer()), encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def setUp(self):
        patcher = mock.patch.object(openai_server, "ARCH", "glm")
        patcher.start()
        self.addCleanup(patcher.stop)
        env = dict(os.environ, CTX="512", COLI_TEMP="0", OMP_NUM_THREADS="2", COLI_NO_OMP_TUNE="1")
        engine = openai_server.Engine(str(BINARY), str(self.snap), cap=8, env=env, family=family_by_id("glm"))
        self.addCleanup(engine.close)
        self.server = APIServer(("127.0.0.1", 0), engine, "glm-tiny")
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def post(self, path, body):
        request = Request(self.base + path, data=json.dumps(body).encode(),
                          headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=60) as response:   # it used to wait forever
            return json.loads(response.read())

    def test_questions_are_answered(self):
        out = self.post("/v1/systemone", {
            "model": "glm-tiny", "state": "My Stripe integration fails and I lose sales.",
            "questions": {
                "urgent": {"type": "noul", "instructions": "Is it urgent?"},
                "team": {"type": "choice", "instructions": "Which team?",
                         "criteria": {"billing": "payments", "technical": "bugs", "sales": "plans"}}}})
        team = out["answers"]["team"]
        self.assertIn(team["choice"], ("billing", "technical", "sales"))
        self.assertAlmostEqual(sum(team["probabilities"].values()), 1.0, places=5)
        self.assertTrue(0.0 <= out["answers"]["urgent"]["noul"] <= 1.0)


@unittest.skipUnless(BINARY, "colibri is not built")
@unittest.skipUnless(FIXTURE.name and (FIXTURE / "model.safetensors").is_file(),
                     "COLI_ORACLE_FIXTURE not set to the tiny GLM oracle (tools/make_glm_oracle.py)")
class GlmBadHeader(unittest.TestCase):
    def test_a_header_it_cannot_read_is_answered_by_its_id(self):
        tmp = tempfile.mkdtemp(prefix="glm_hdr_")
        self.addCleanup(shutil.rmtree, tmp, True)
        snap = Path(tmp) / "snap"
        shutil.copytree(FIXTURE, snap)
        (snap / "tokenizer.json").write_text(json.dumps(byte_tokenizer()), encoding="utf-8")
        body = b"Context:\nhello\n\nIs it urgent?"
        good = f"SUBMIT 8 0 {len(body)} 2 0 1 0\n".encode() + body + b"\n"
        bad = f"SUBMIT 7 0 {len(body)} 0 0 1 logprobs=1 pin=1\n".encode() + body + b"\n"   # no grammar field
        env = dict(os.environ, SNAP=str(snap), SERVE="1", SERVE_BATCH="1", KV_SLOTS="1", CTX="256",
                   COLI_TEMP="0", OMP_NUM_THREADS="2", COLI_NO_OMP_TUNE="1")
        out = subprocess.run([str(BINARY), "0"], input=bad + good, env=env, capture_output=True, timeout=120).stdout
        lines = [line for line in out.split(b"\n") if line.startswith((b"ERROR", b"DONE", b"ACCEPT"))]
        self.assertIn(b"ERROR 7 BAD_REQUEST", lines, out[-600:])
        self.assertNotIn(b"ERROR 0 BAD_REQUEST", lines, "the payload was read as commands")
        self.assertTrue(any(line.startswith(b"DONE 8 ") for line in lines), "the next request was not served")


if __name__ == "__main__":
    unittest.main()
