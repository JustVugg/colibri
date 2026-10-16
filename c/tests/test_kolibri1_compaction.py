"""Kolibri-1 through the gateway's context compaction: a mock engine with a small context.

The compaction loop and its summary request are shared by every family; Kolibri-1's template
reads the client's raw `enable_thinking` (and lets `reasoning_effort` win), so the gateway
hands the loop's renderer that raw value for this family only. This test drives the real
gateway over HTTP (model_type "kolibri1", the real registry and renderer) against a mock
engine that refuses prompts over a small capacity, so compaction is forced, and checks:

  * default thinking, explicit `enable_thinking: false`, and conflicting effort/switch values
    (effort wins either way) each produce the turn's prompt that the pinned chat template
    renders for that history and those settings, byte for byte (needs jinja2);
  * the summary request is the same history plus the summary question, rendered with the
    reasoning block closed, byte for byte against the template;
  * the stored summary carries the summary text only. The mock writes a <think> block before
    every answer unless the prompt already closed the reasoning block; the summary request
    closes it, so no reasoning may reach the summary, and the turn's own reasoning must come
    back as `reasoning_content`, not in the stored summary or the content;
  * the summary replaces the old turns in the re-rendered conversation.

CPU-light: no model, no numpy. Skipped on non-POSIX hosts (the mock is a shebang script).
"""
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import openai_server as gateway  # noqa: E402

try:
    import test_kolibri1_chat_template as pinned     # the template, rendered by jinja2
    HAVE_TEMPLATE = pinned.jinja2 is not None
except Exception:                                      # pragma: no cover
    pinned, HAVE_TEMPLATE = None, False

SERVER = HERE.parent / "openai_server.py"
CAPACITY = 700            # tokens, a token per four bytes
SUMMARY = "SUMMARY: the user listed topics one to five."
REASONING = "planning the summary in private"

MOCK_ENGINE = r'''#!/usr/bin/env python3
import json, os, sys
out, inp = sys.stdout.buffer, sys.stdin.buffer
cap = int(os.environ["MOCK_CTX"])
out.write(b"\x01\x01READY\x01\x01\n" + b"CAPS vision=0\n" + b"STAT 0 0 0 0 0\n"); out.flush()

def log(entry):
    with open(os.environ["MOCK_LOG"], "a") as f:
        f.write(json.dumps(entry) + "\n")

while True:
    line = inp.readline()
    if not line: break
    f = line.decode().strip().split()
    if not f or f[0] != "SUBMIT": continue
    rid, plen, max_tok = f[1], int(f[3]), int(f[4])
    prompt = inp.read(plen).decode("utf-8", "replace"); inp.read(1)
    tokens = (len(prompt.encode()) + 3) // 4
    if tokens >= cap:
        log({"prompt": prompt, "tokens": tokens, "refused": True})
        out.write(("ERROR %s CONTEXT_EXCEEDED prompt_tokens=%d requested=%d capacity=%d\n"
                   % (rid, tokens, max_tok, cap)).encode()); out.flush()
        continue
    log({"prompt": prompt, "tokens": tokens, "refused": False})
    out.write(("ACCEPT %s %d\n" % (rid, tokens)).encode()); out.flush()
    closed = prompt.endswith("<think>\n\n</think>\n\n")      # the prompt already closed the reasoning block
    ask = "Summarize our conversation so far" in prompt
    body = SUMMARY if ask else "Noted, turn answered."
    # a model writes reasoning first unless the prompt closed the block
    text = body if closed else "<think>\nREASON\n</think>\n\n" + body
    data = text.encode()
    out.write(("DATA %s %d\n" % (rid, len(data))).encode() + data + b"\n")
    out.write(("DONE %s STAT %d 1.0 50.0 1.0 %d 0\n" % (rid, len(text.split()), tokens)).encode())
    out.flush()
'''.replace("REASON\\n", REASONING + "\\n").replace("SUMMARY", repr(SUMMARY))


def history(tag, turns=6):
    messages = [{"role": "system", "content": f"You are a helpful assistant ({tag})."}]
    for i in range(turns):
        messages.append({"role": "user",
                         "content": f"Topic {i} {tag}: " + ("lorem ipsum dolor sit amet " * 10).strip()})
        messages.append({"role": "assistant", "content": f"Reply {i}: " + "noted " * 40})
    messages.append({"role": "user", "content": f"What was the last topic ({tag})?"})
    return messages


@unittest.skipUnless(os.name == "posix", "the mock engine is a shebang script")
class Kolibri1Compaction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        (root / "config.json").write_text(json.dumps({"model_type": "kolibri1"}), encoding="utf-8")
        mock = root / "mock_engine.py"
        mock.write_text(MOCK_ENGINE)
        mock.chmod(0o755)
        cls.log = root / "prompts.jsonl"
        cls.log.touch()
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            cls.port = probe.getsockname()[1]
        env = dict(os.environ, MOCK_LOG=str(cls.log), MOCK_CTX=str(CAPACITY), CTX=str(CAPACITY))
        for name in ("COLI_API_KEY", "COLI_COMPACT", "COLI_THINK"):
            env.pop(name, None)
        cls.err = open(root / "gateway.err", "wb")
        cls.server = subprocess.Popen(
            [sys.executable, str(SERVER), "--model", str(root), "--engine", str(mock),
             "--port", str(cls.port), "--cap", "8", "--max-tokens", "64"],
            env=env, stdout=subprocess.DEVNULL, stderr=cls.err)
        cls.base = f"http://127.0.0.1:{cls.port}/v1"
        for _ in range(150):
            try:
                with urllib.request.urlopen(cls.base + "/models", timeout=2) as resp:
                    cls.model = json.loads(resp.read())["data"][0]["id"]
                    return
            except OSError:
                if cls.server.poll() is not None:
                    raise RuntimeError("gateway exited during startup:\n"
                                       + (root / "gateway.err").read_text(errors="replace")[-2000:])
                time.sleep(0.1)
        raise RuntimeError("gateway did not come up")

    @classmethod
    def tearDownClass(cls):
        cls.server.terminate()
        try:
            cls.server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            cls.server.kill()
        cls.err.close()
        cls.tmp.cleanup()

    def prompts(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    def compact(self, messages, **settings):
        """(response, prompts the engine saw for this request)"""
        before = len(self.prompts())
        body = dict(settings, model=self.model, messages=messages, max_tokens=16,
                    context_compaction="auto")
        req = urllib.request.Request(self.base + "/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            reply = json.loads(resp.read())
        return reply, self.prompts()[before:]

    def expected_turn(self, messages, reply, **settings):
        """What the pinned template renders for the summarized history and these settings."""
        done = reply["compaction"]["summarized_messages"]
        lead, conversation = gateway.split_lead(messages)
        shown = gateway.with_summary(lead, reply["compaction"]["summary"]) + conversation[done:]
        return pinned.reference_render({"messages": shown}, **settings)

    def check_scenario(self, tag, settings, thinking_sentence_on):
        messages = history(tag)
        reply, seen = self.compact(messages, **settings)
        message = reply["choices"][0]["message"]
        self.assertEqual(message["content"].strip(), "Noted, turn answered.")
        if thinking_sentence_on:
            self.assertEqual((message.get("reasoning_content") or "").strip(), REASONING)
        else:
            self.assertFalse(message.get("reasoning_content"))
        asks = [p for p in seen if "Summarize our conversation so far" in p["prompt"]]
        self.assertTrue(asks, "compaction was not forced")
        final = seen[-1]
        self.assertFalse(final["refused"])
        # the stored summary is the summary text: the reasoning written before it is gone
        stored = reply["compaction"]["summary"]
        self.assertEqual(stored, SUMMARY)
        self.assertNotIn(REASONING, json.dumps(reply["compaction"]))
        self.assertNotIn("<think>", stored)
        # the summary replaced the old turns, in the system message
        self.assertIn(SUMMARY, final["prompt"])
        self.assertNotIn(f"Topic 0 {tag}", final["prompt"])
        self.assertNotIn(REASONING, final["prompt"])
        self.assertIn(f"What was the last topic ({tag})?", final["prompt"])
        # the turn is rendered with the intended settings
        sentence_on = "Reasoning effort is set to" in final["prompt"]
        disabled = gateway.KOLIBRI1_SENTENCES["none"] in final["prompt"]
        self.assertEqual(sentence_on, thinking_sentence_on)
        self.assertEqual(disabled, not thinking_sentence_on)
        self.assertEqual(final["prompt"].endswith("<think>\n\n</think>\n\n"), not thinking_sentence_on)
        for ask in asks:
            self.assertFalse(ask["refused"], "a summary request that did not fit was sent")
            self.assertNotIn(f"What was the last topic ({tag})?", ask["prompt"])
            # the reasoning block is closed in the summary request, whatever the turn's setting
            self.assertTrue(ask["prompt"].endswith("<think>\n\n</think>\n\n"), ask["prompt"][-80:])
        if HAVE_TEMPLATE:
            self.assertEqual(final["prompt"], self.expected_turn(messages, reply, **{
                k: v for k, v in settings.items() if k in ("reasoning_effort", "enable_thinking")}))
            if len(asks) == 1:                               # one range: the history, then the question
                prompt = asks[0]["prompt"]
                head, _, tail = prompt.rpartition("<|im_start|>user\n")
                question = tail.split("<|im_end|>")[0]
                lead, conversation = gateway.split_lead(messages)
                end = max(i for i, m in enumerate(conversation) if m["role"] == "user")
                shown = lead + conversation[:end] + [{"role": "user", "content": question}]
                self.assertEqual(prompt, pinned.reference_render(
                    {"messages": shown}, enable_thinking=False))
        return reply, final

    def test_default_thinking_is_on(self):
        self.check_scenario("default", {}, thinking_sentence_on=True)

    def test_explicit_thinking_disabled(self):
        self.check_scenario("off", {"enable_thinking": False}, thinking_sentence_on=False)

    def test_effort_beats_a_conflicting_switch_to_disable(self):
        # effort "none" turns thinking off although the switch says on
        self.check_scenario("none-vs-on", {"reasoning_effort": "none", "enable_thinking": True},
                            thinking_sentence_on=False)

    def test_effort_beats_a_conflicting_switch_to_enable(self):
        # effort "high" keeps thinking on although the switch says off
        _, final = self.check_scenario("high-vs-off", {"reasoning_effort": "high",
                                                       "enable_thinking": False},
                                       thinking_sentence_on=True)
        self.assertIn(gateway.KOLIBRI1_SENTENCES["high"], final["prompt"])

    def test_a_lower_effort_is_kept_in_the_re_rendered_turn(self):
        _, final = self.check_scenario("low", {"reasoning_effort": "low"}, thinking_sentence_on=True)
        self.assertIn(gateway.KOLIBRI1_SENTENCES["low"], final["prompt"])


if __name__ == "__main__":
    unittest.main()
