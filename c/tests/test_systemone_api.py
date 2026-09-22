"""POST /v1/systemone: the request and the reply of TypeSafe's Jev API, served
by the brio channel.

A client written for Jev sends `state`, `model` and a map of questions typed
noul / choice / score, and reads `answers` keyed by its own ids plus
`usage.input_tokens/output_tokens`. This pins the mapping onto the `questions`
form of /v1/brio with the deterministic scoring engine of test_brio_api: what
each primitive puts in the prompt, what comes back, the confidence formula
their docs give, that any model name is accepted on this route, and that the
state is still photographed once for all the questions.
"""
import json
import math
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from openai_server import APIServer
from tests.test_brio_api import ScoringEngine

STATE = ("Hi, I've been trying to connect my Stripe account for 3 days and the "
         "integration keeps failing. I'm losing sales. Please help ASAP.")


def confidence(ps):
    n = len(ps)
    return (n * max(ps) - 1.0) / (n - 1)


class SystemOneApi(unittest.TestCase):
    def serve(self, table):
        self.engine = ScoringEngine(table)
        self.server = APIServer(("127.0.0.1", 0), self.engine, "test-model")
        self.addCleanup(self.server.shutdown)
        self.addCleanup(self.server.server_close)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def post(self, body):
        request = Request(self.base + "/v1/systemone", data=json.dumps(body).encode(),
                          headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=10) as response:
            return json.loads(response.read())

    def post_error(self, body):
        try:
            self.post(body)
        except HTTPError as error:
            return error.code, json.loads(error.read())
        self.fail("expected an error")

    def pins(self):
        return [c["prompt"] for c in self.engine.calls if c["pin"]]

    def test_noul_is_the_probability_of_yes(self):
        self.serve({"yes": -0.1, "no": -2.0})
        out = self.post({"model": "jev-latest", "state": STATE, "questions": {
            "urgency": {"type": "noul", "instructions": "Does this message express urgency?"}}})
        answer = out["answers"]["urgency"]
        self.assertEqual(answer["type"], "noul")
        want = math.exp(-0.1) / (math.exp(-0.1) + math.exp(-2.0))
        self.assertAlmostEqual(answer["noul"], want, places=5)
        self.assertNotIn("confidence", answer)          # their docs: noul carries none
        # the question is asked as a yes/no question, the state as the context
        asked = [c["prompt"] for c in self.engine.calls if "Does this message" in c["prompt"]]
        self.assertTrue(asked and "Answer yes or no." in asked[0], asked[:1])

    def test_noul_criteria_go_into_the_question(self):
        self.serve({"yes": -0.1, "no": -2.0})
        self.post({"model": "jev-latest", "state": STATE, "questions": {
            "q": {"type": "noul", "instructions": "Is the customer at risk of churning?",
                  "criteria": {"true": "they threaten to leave or mention losses",
                               "false": "a routine question"}}}})
        pinned = self.pins()[1]
        self.assertIn("yes: they threaten to leave", pinned)
        self.assertIn("no: a routine question", pinned)

    def test_choice_labels_probabilities_and_confidence(self):
        self.serve({"billing": -0.2, "technical": -3.0, "sales": -4.0})
        out = self.post({"model": "jev-latest", "state": STATE, "questions": {
            "department": {"type": "choice", "instructions": "Which team should handle this?",
                           "criteria": {"billing": "payments, invoices, Stripe",
                                        "technical": "bugs and outages",
                                        "sales": "pricing and plans"}}}})
        answer = out["answers"]["department"]
        self.assertEqual(answer["type"], "choice")
        self.assertEqual(answer["choice"], "billing")
        self.assertEqual(list(answer["probabilities"]), ["billing", "technical", "sales"])
        self.assertAlmostEqual(sum(answer["probabilities"].values()), 1.0, places=5)
        self.assertAlmostEqual(answer["confidence"],
                               confidence(list(answer["probabilities"].values())), places=5)
        # the descriptions are in the question, the labels are the options scored
        pinned = self.pins()[1]
        self.assertIn("- billing: payments, invoices, Stripe", pinned)
        self.assertIn("- sales: pricing and plans", pinned)
        scored = [c["prompt"] for c in self.engine.calls if not c["pin"]]
        self.assertTrue(any(p.endswith(" technical") for p in scored), scored[-3:])

    def test_choice_with_null_descriptions_still_lists_the_labels(self):
        self.serve({"a": -0.1, "b": -1.0})
        out = self.post({"model": "jev-latest", "state": STATE, "questions": {
            "q": {"type": "choice", "criteria": {"a": None, "b": None}}}})
        self.assertEqual(out["answers"]["q"]["choice"], "a")
        self.assertIn("- a\n- b", self.pins()[1])

    def test_score_expected_value_legend_and_confidence(self):
        self.serve({"4": -0.1, "1": -5.0, "2": -5.0, "3": -5.0, "5": -5.0})
        levels = ["not urgent", "low", "moderate", "high", "critical"]
        out = self.post({"model": "jev-latest", "state": STATE, "questions": {
            "urgency": {"type": "score", "instructions": "How urgent is this?", "criteria": levels}}})
        answer = out["answers"]["urgency"]
        self.assertEqual(answer["type"], "score")
        self.assertEqual(answer["legend"], {str(i + 1): d for i, d in enumerate(levels)})
        self.assertEqual(list(answer["probabilities"]), ["1", "2", "3", "4", "5"])
        ps = answer["probabilities"]
        self.assertAlmostEqual(sum(ps.values()), 1.0, places=5)
        self.assertAlmostEqual(answer["score"], sum(int(k) * v for k, v in ps.items()), places=5)
        self.assertGreater(answer["score"], 3.9)             # the mass sits on level 4
        self.assertAlmostEqual(answer["confidence"], confidence(list(ps.values())), places=5)
        self.assertIn("4: high", self.pins()[1])

    def test_structured_state_and_instructions_are_serialized(self):
        self.serve({"yes": -0.1, "no": -2.0})
        out = self.post({"model": "jev-latest",
                         "state": {"ticket": 4711, "text": "refund not received"},
                         "questions": {"q": {"type": "noul",
                                             "instructions": {"ask": "is this about money?"}}}})
        self.assertEqual(out["model"], "test-model")       # the served model answers
        state_pin = self.pins()[0]
        self.assertIn('"ticket": 4711', state_pin)
        self.assertIn('"ask": "is this about money?"', self.pins()[1])

    def test_state_is_photographed_once_for_all_questions(self):
        self.serve({"yes": -0.1, "no": -2.0, "a": -0.1, "b": -1.0})
        out = self.post({"model": "jev-latest", "state": STATE, "questions": {
            "one": {"type": "noul", "instructions": "q1?"},
            "two": {"type": "choice", "criteria": {"a": "x", "b": "y"}},
            "three": {"type": "score", "criteria": ["bad", "good"]}}})
        self.assertEqual(list(out["answers"]), ["one", "two", "three"])
        pins = self.pins()
        self.assertEqual(len(pins), 4, pins)               # the state, then each question once
        self.assertEqual(pins[0], f"Context:\n{STATE}\n\n")
        self.assertEqual(set(out["usage"]), {"input_tokens", "output_tokens"})
        self.assertGreater(out["usage"]["input_tokens"], 0)
        self.assertGreater(out["usage"]["output_tokens"], 0)

    def test_validation_errors_are_422(self):
        self.serve({})
        code, body = self.post_error({"model": "jev-latest", "questions": {"q": {"type": "noul"}}})
        self.assertEqual(code, 422)
        self.assertIn("state", body["error"]["message"])
        code, body = self.post_error({"model": "jev-latest", "state": STATE,
                                      "questions": {"q": {"type": "guess"}}})
        self.assertEqual(code, 422)
        self.assertIn("type", body["error"]["message"])
        code, body = self.post_error({"model": "jev-latest", "state": STATE,
                                      "questions": {"q": {"type": "choice", "criteria": ["a", "b"]}}})
        self.assertEqual(code, 422)
        code, body = self.post_error({"model": "jev-latest", "state": STATE,
                                      "questions": {"q": {"type": "score", "criteria": ["only one"]}}})
        self.assertEqual(code, 422)
        self.assertIn("2 to 10", body["error"]["message"])
        code, body = self.post_error({"model": "jev-latest", "state": STATE, "questions": []})
        self.assertEqual(code, 422)

    def test_brio_route_still_checks_the_model_name(self):
        self.serve({"a": -0.1, "b": -1.0})
        request = Request(self.base + "/v1/brio",
                          data=json.dumps({"model": "jev-latest", "state": STATE,
                                           "options": ["a", "b"]}).encode(),
                          headers={"Content-Type": "application/json"})
        with self.assertRaises(HTTPError) as caught:
            urlopen(request, timeout=10)
        self.assertEqual(caught.exception.code, 404)


if __name__ == "__main__":
    unittest.main()
