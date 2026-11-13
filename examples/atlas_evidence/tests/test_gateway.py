"""Actual upstream HTTP gateway with its scripted engine double, without weights.

These are protocol interoperability checks, not live model-quality measurements.
Upstream uses bare module imports from c/. Each check runs in a short-lived Python
process so those import paths and module names never alter the enclosing suite.
"""

import json
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]

SCRIPT = r'''
import importlib.util
import json
from pathlib import Path
import sys
import threading

root = Path(sys.argv[1])
scenario = sys.argv[2]
sys.path.insert(0, str(root / "c"))
sys.path.insert(1, str(root))

# Load the existing upstream engine double by a unique module name rather than
# importing a generic tests package. No engine executable or weights are loaded.
spec = importlib.util.spec_from_file_location(
    "atlas_upstream_gateway_fixtures", root / "c/tests/test_openai_server.py")
upstream_tests = importlib.util.module_from_spec(spec)
spec.loader.exec_module(upstream_tests)
from openai_server import APIServer
from examples.atlas_evidence.client import ClientError, build_request, generate
from examples.atlas_evidence.core import assess, build_capture, snapshot

text = "Synthetic policy café 🐦\nReporting is annual.\n"
quote = "Reporting is annual."
start = text.index(quote)
candidate = {"claims": [{"id": "claim_annual", "text": "The synthetic policy says reporting is annual.",
                         "source_id": "policy", "quote": quote,
                         "start": start, "end": start + len(quote)}]}
content = json.dumps(candidate, ensure_ascii=False)
engine = upstream_tests.ScriptedEngine(chunks=(content,), length_limited=scenario == "truncated")
server = APIServer(("127.0.0.1", 0), engine, "atlas-gateway-fixture",
                   api_key="fixture-secret", max_tokens=4096)
thread = threading.Thread(target=server.serve_forever, args=(.01,), daemon=True)
thread.start()
try:
    sources = [snapshot("policy", text, title="Synthetic policy")]
    request = build_request(sources, "atlas-gateway-fixture", max_tokens=2048)
    endpoint = f"http://127.0.0.1:{server.server_port}/v1"
    denied = None
    calls_after_denial = None
    if scenario == "authentication":
        try:
            generate(request, endpoint=endpoint, api_key="wrong-secret", timeout=5)
        except ClientError as exc:
            denied = str(exc)
        calls_after_denial = len(engine.calls)
    response = generate(request, endpoint=endpoint, api_key="fixture-secret", timeout=5)
    capture = build_capture(sources, request, response, mode="fixture",
                            model_revision="ScriptedEngine-test-double-no-weights",
                            engine_revision="upstream-openai-server-source-checkout")
    report = assess(capture)
    print(json.dumps({"response": response, "report": report,
                      "mode": capture["mode"], "engine_calls": len(engine.calls),
                      "engine_received_json_grammar": bool(engine.calls[0][5]),
                      "engine_received_source": request["messages"][1]["content"] in engine.calls[0][0],
                      "capture_contains_api_key": "fixture-secret" in json.dumps(capture),
                      "denied": denied, "calls_after_denial": calls_after_denial}, ensure_ascii=False))
finally:
    server.scheduler.close()
    server.shutdown()
    server.server_close()
    thread.join(2)
'''


class UpstreamGatewayTests(unittest.TestCase):
    def run_gateway(self, scenario):
        result = subprocess.run([sys.executable, "-c", SCRIPT, str(ROOT), scenario],
                                cwd=ROOT, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_complete_candidate_through_actual_gateway(self):
        result = self.run_gateway("complete")
        self.assertEqual(result["response"]["model"], "atlas-gateway-fixture")
        self.assertEqual(result["response"]["choices"][0]["finish_reason"], "stop")
        self.assertEqual(result["engine_calls"], 1)
        self.assertTrue(result["engine_received_json_grammar"])
        self.assertTrue(result["engine_received_source"])
        self.assertTrue(result["report"]["passed"])
        self.assertEqual(result["report"]["semantic_status"], "unreviewed")
        self.assertEqual(result["mode"], "fixture")
        self.assertFalse(result["capture_contains_api_key"])

    def test_gateway_authentication_precedes_engine_work(self):
        result = self.run_gateway("authentication")
        self.assertEqual(result["denied"], "Local server returned HTTP 401")
        self.assertEqual(result["calls_after_denial"], 0)
        self.assertEqual(result["engine_calls"], 1)
        self.assertTrue(result["report"]["passed"])
        self.assertFalse(result["capture_contains_api_key"])

    def test_gateway_length_finish_is_not_accepted_as_complete(self):
        # The double supplies syntactically complete JSON but reports an exhausted
        # generation budget. The adapter must honour that signal and reject it.
        result = self.run_gateway("truncated")
        self.assertEqual(result["response"]["choices"][0]["finish_reason"], "length")
        self.assertEqual(result["engine_calls"], 1)
        self.assertFalse(result["report"]["passed"])
        self.assertTrue(any("incomplete" in error for error in result["report"]["errors"]))


if __name__ == "__main__":
    unittest.main()
