#!/usr/bin/env python3
"""A stand-in for the kolibri1 engine's SERVE loop, for gateway end-to-end tests.

kolibri1.c has a SERVE loop (tests/test_kolibri1_gateway_real_engine.py drives the
real binary); this stub is kept because it answers as a function of the rendered prompt, so a test can
read the exact prompt text and the gateway's splitter / parser result without a model. It speaks the same stdin/stdout protocol
(docs/serve_protocol.md: READY, SUBMIT frames, ACCEPT, DATA, DONE) so that the REAL gateway
(`openai_server.py --model <tiny fixture> --engine <this>`), the REAL family lookup and the
REAL launch path (model_type -> family_registry -> cap_for_arch -> Engine) can be driven over
HTTP. It does no model math: the reply is a function of the rendered prompt, so a test can see
which prompt the gateway sent and what the splitter and tool parser made of the answer.

It writes one JSON line per request to $KOLIBRI1_STUB_LOG (prompt, argv, a few env values).
"""
import json
import os
import re
import sys

READY = b"\x01\x01READY\x01\x01\n"
OUT = sys.stdout.buffer
IN = sys.stdin.buffer


def send(data):
    OUT.write(data)
    OUT.flush()


def last_user_turn(prompt):
    turns = re.findall(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", prompt, re.S)
    return turns[-1] if turns else ""


def reply_for(prompt):
    effort = "high"
    match = re.search(r"# Reasoning effort\n\n(Reasoning[^.]*\.)", prompt)
    sentence = match.group(1) if match else ""
    if sentence.startswith("Reasoning is disabled"):
        effort = "none"
    elif "set to low" in sentence:
        effort = "low"
    elif "set to medium" in sentence:
        effort = "medium"
    thinking = prompt.endswith("<|im_start|>assistant\n")
    off_cue = prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    open_turn = not (thinking or off_cue)        # the prompt stops inside an assistant turn
    user = last_user_turn(prompt)
    tools = "<tools>" in prompt
    if open_turn:
        return " und dann weiter – übermäßig schön \U0001f600"
    if tools and user.startswith("<tool_response>"):
        body = "Es sind 12 Grad in Köln."
    elif tools and "Wetter" in user:
        body = ("Ich schaue nach.\n<tool_call>\n"
                '{"name": "get_weather", "arguments": {"city": "Köln"}}\n</tool_call>')
    else:
        body = f"Antwort ({effort}): Grüße aus München \U0001f600"
    if thinking:
        return f"<think>\nPrüfe die Aufgabe, Aufwand {effort}.\n</think>\n\n" + body
    return body


def main():
    send(READY + b"CAPS chat=1\n" + b"STAT 0 0 0 0\n")
    log = os.environ.get("KOLIBRI1_STUB_LOG")
    while True:
        line = IN.readline()
        if not line:
            return
        fields = line.decode("utf-8", "replace").split()
        if not fields or fields[0] != "SUBMIT":
            continue                        # STOP / CANCEL / PIN ... : nothing to stop here
        request_id, size = fields[1], int(fields[3])
        prompt = b""
        while len(prompt) < size:
            prompt += IN.read(size - len(prompt))
        IN.read(1)                          # the frame's trailing newline
        text = prompt.decode("utf-8")
        reply = reply_for(text)
        if log:
            with open(log, "a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "argv": sys.argv[1:], "header": fields[:7], "prompt": text,
                    "env": {key: os.environ.get(key) for key in
                            ("SNAP", "SERVE", "SERVE_BATCH", "CTX", "KV_SLOTS", "NGEN")}},
                    ensure_ascii=False) + "\n")
        send(f"ACCEPT {request_id} {max(1, len(text) // 4)}\n".encode())
        chars = list(reply)
        for i in range(0, len(chars), 3):   # whole characters, as the engine's decoder emits
            chunk = "".join(chars[i:i + 3]).encode("utf-8")
            send(f"DATA {request_id} {len(chunk)}\n".encode() + chunk + b"\n")
        send(f"DONE {request_id} STAT {len(chars)} 2.500 0.0 1.250 {max(1, len(text) // 4)} 0\n".encode())


if __name__ == "__main__":
    main()
