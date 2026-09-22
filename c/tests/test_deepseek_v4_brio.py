#!/usr/bin/env python3
"""The numeric channel on the DeepSeek V4 serve path (SUBMIT logprobs=k / pin=1).

This is what `POST /v1/brio` speaks (docs/brio.md), and #1648 found that this
engine accepted the header and then refused the request. The properties under
test are the exact ones a closed-set caller relies on:

- a read-only request (max_tokens=0, logprobs>0) answers with one ECHO frame
  per prompt position after the first, and generates nothing;
- the ECHO at position p is the head over prompt[:p]: its best token is the
  token a greedy generation from that same prefix produces;
- a pinned prompt gives every prompt that extends it the predictor of its
  first fresh token, from the kept scores, and those scores are the ones a
  cold prefill computes;
- with the channel open, generation is unchanged and every DATA frame carries
  the tail of the token it holds.

Speaks SUBMIT/ECHO/DATA/DONE directly, as tests/test_deepseek_v4_prefix.py does.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path


def token_prompt(ids: list[int]) -> str:
    return "".join(f"<t{token:03d}>" for token in ids)


def parse_tokens(text: str) -> list[int]:
    return [int(match) for match in re.findall(r"<t(\d+)>", text)]


class Reply:
    def __init__(self) -> None:
        self.accepted = -1
        self.echoes: dict[int, dict] = {}      # position -> {"token", "lp", "top": {tid: tlp}}
        self.data: list[tuple[bytes, dict | None]] = []
        self.reuse = -1
        self.completion = -1

    @property
    def text(self) -> str:
        return b"".join(piece for piece, _ in self.data).decode("utf-8", "replace")


def parse_tail(fields: list[str]) -> dict | None:
    """`<lp> <k> [tid tlp]*k` -> {"lp": float, "top": {tid: tlp}}; None for no tail."""
    if not fields:
        return None
    lp = None if fields[0] == "nan" else float(fields[0])
    k = int(fields[1])
    pairs = fields[2:2 + 2 * k]
    top = {int(pairs[i]): float(pairs[i + 1]) for i in range(0, len(pairs), 2)}
    return {"lp": lp, "top": top}


class Serve:
    """One persistent `SERVE=1` engine process."""

    def __init__(self, binary: Path, model: Path, ctx: str = "128") -> None:
        env = dict(os.environ, SERVE="1", SNAP=str(model), CTX=ctx)
        self.process = subprocess.Popen(
            [str(binary)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, env=env,
        )
        self.counter = 0

    def submit(self, prompt: str, max_tokens: int, logprobs: int = 0,
               pin: bool = False) -> Reply:
        self.counter += 1
        request_id = f"r{self.counter}"
        payload = prompt.encode("utf-8")
        header = f"SUBMIT {request_id} 0 {len(payload)} {max_tokens} 0.0 1.0 0"
        if logprobs:
            header += f" logprobs={logprobs}"
        if pin:
            header += " pin=1"
        assert self.process.stdin is not None
        self.process.stdin.write(header.encode("ascii") + b"\n" + payload + b"\n")
        self.process.stdin.flush()

        reply = Reply()
        stream = self.process.stdout
        assert stream is not None
        while True:
            line = stream.readline()
            if not line:
                raise AssertionError(
                    f"engine closed stdout during {request_id}; "
                    f"stderr:\n{self._drain_stderr()}")
            fields = line.decode("utf-8", "replace").split()
            if not fields:
                continue
            kind = fields[0]
            if kind == "ERROR":
                raise AssertionError(f"engine ERROR: {line!r}")
            if kind == "ACCEPT":
                reply.accepted = int(fields[2])
            elif kind == "DATA":
                size = int(fields[2])
                piece = stream.read(size)
                stream.read(1)
                reply.data.append((piece, parse_tail(fields[3:])))
            elif kind == "ECHO":
                # ECHO <id> <n> <pos> <lp> <k> [tid tlp]*k, then n bytes + LF
                size = int(fields[2])
                piece = stream.read(size)
                stream.read(1)
                tail = parse_tail(fields[4:])
                assert tail is not None
                reply.echoes[int(fields[3])] = {"token": piece.decode("utf-8", "replace"), **tail}
            elif kind == "DONE":
                # DONE <id> STAT <completion> <tok/s> <hit%> <rss> <prompt> <cap> [<reuse>]
                reply.completion = int(fields[3])
                if len(fields) >= 10:
                    reply.reuse = int(fields[9])
                break
        return reply

    def _drain_stderr(self) -> str:
        assert self.process.stderr is not None
        try:
            return self.process.stderr.read(8192).decode("utf-8", "replace")
        except Exception:                          # pragma: no cover
            return "<unavailable>"

    def close(self) -> None:
        try:
            if self.process.stdin:
                self.process.stdin.close()
            self.process.wait(timeout=30)
        except Exception:                          # pragma: no cover
            self.process.kill()


def best_of(echo: dict) -> int:
    return max(echo["top"], key=lambda tid: echo["top"][tid])


def check_read_only(binary: Path, model: Path, ids: list[int]) -> None:
    """max_tokens=0 with logprobs: one ECHO per position after the first, no DATA."""
    serve = Serve(binary, model)
    try:
        reply = serve.submit(token_prompt(ids), 0, logprobs=3)
    finally:
        serve.close()
    n = len(ids)
    if reply.accepted != n:
        raise AssertionError(f"ACCEPT said {reply.accepted} tokens for a {n}-token prompt")
    if reply.data:
        raise AssertionError(f"a read-only request generated {len(reply.data)} DATA frames")
    if reply.completion != 0:
        raise AssertionError(f"DONE reported {reply.completion} completion tokens")
    if sorted(reply.echoes) != list(range(1, n)):
        raise AssertionError(f"ECHO positions {sorted(reply.echoes)}, expected 1..{n - 1}")
    for position, echo in reply.echoes.items():
        if parse_tokens(echo["token"]) != [ids[position]]:
            raise AssertionError(f"ECHO at {position} names {echo['token']!r}, prompt has {ids[position]}")
        if echo["lp"] is None or echo["lp"] > 1e-6:
            raise AssertionError(f"ECHO at {position}: log-probability {echo['lp']} is not <= 0")
        if len(echo["top"]) != 3:
            raise AssertionError(f"ECHO at {position}: asked top-3, got {len(echo['top'])}")
        mass = sum(math.exp(v) for v in echo["top"].values())
        if mass > 1.0 + 1e-4:
            raise AssertionError(f"ECHO at {position}: top-3 mass {mass} exceeds 1")
        if ids[position] in echo["top"] and abs(echo["top"][ids[position]] - echo["lp"]) > 1e-5:
            raise AssertionError(f"ECHO at {position}: the token's own lp differs from its top-k entry")
    print(f"PASS read-only: {n - 1} ECHO frames, zero generated tokens")


def check_echo_is_the_head(binary: Path, model: Path, ids: list[int]) -> None:
    """The best token of the ECHO at p is what greedy decoding emits after prompt[:p]."""
    serve = Serve(binary, model)
    try:
        full = serve.submit(token_prompt(ids), 0, logprobs=4)
        for position in sorted({1, len(ids) // 2, len(ids) - 1}):
            cold = Serve(binary, model)
            try:
                gen = cold.submit(token_prompt(ids[:position]), 1)
            finally:
                cold.close()
            produced = parse_tokens(gen.text)
            expected = best_of(full.echoes[position])
            if produced != [expected]:
                raise AssertionError(
                    f"position {position}: ECHO's best token is {expected}, "
                    f"greedy generation from the same prefix gave {produced}")
    finally:
        serve.close()
    print("PASS echo is the head: best ECHO token == greedy token from the same prefix")


def check_pin(binary: Path, model: Path, ids: list[int], a: int, b: int) -> None:
    """Pin the prompt, ask two options: both start from the pin with the first token's predictor."""
    n = len(ids)
    serve = Serve(binary, model)
    try:
        pinned = serve.submit(token_prompt(ids), 0, logprobs=2, pin=True)
        if pinned.completion != 0 or pinned.data:
            raise AssertionError("the pinned read generated tokens")
        first = serve.submit(token_prompt(ids + [a]), 0, logprobs=2)
        second = serve.submit(token_prompt(ids + [b]), 0, logprobs=2)
    finally:
        serve.close()
    for name, reply in (("first option", first), ("second option", second)):
        if reply.reuse != n:
            raise AssertionError(f"{name}: reused {reply.reuse} tokens, the pin holds {n}")
        if sorted(reply.echoes) != [n]:
            raise AssertionError(f"{name}: ECHO positions {sorted(reply.echoes)}, expected [{n}]")
    if first.echoes[n]["top"] != second.echoes[n]["top"]:
        raise AssertionError("the two options saw different predictors for the same pinned prompt")
    if parse_tokens(first.echoes[n]["token"]) != [a] or parse_tokens(second.echoes[n]["token"]) != [b]:
        raise AssertionError("the ECHO at the option position does not name the option token")
    # the kept scores are what a cold prefill computes at that position
    cold = Serve(binary, model)
    try:
        fresh = cold.submit(token_prompt(ids + [a]), 0, logprobs=2)
    finally:
        cold.close()
    if fresh.reuse != 0:
        raise AssertionError(f"cold engine reported reuse={fresh.reuse}")
    warm_top, cold_top = first.echoes[n]["top"], fresh.echoes[n]["top"]
    if warm_top.keys() != cold_top.keys() or any(abs(warm_top[t] - cold_top[t]) > 1e-4 for t in warm_top):
        raise AssertionError(f"pinned predictor {warm_top} differs from a cold prefill's {cold_top}")
    if abs(first.echoes[n]["lp"] - fresh.echoes[n]["lp"]) > 1e-4:
        raise AssertionError("the option's log-probability differs between the pin and a cold prefill")
    print(f"PASS pin: both options reused {n} tokens and read the pinned predictor, equal to a cold prefill")


def check_generation_with_tail(binary: Path, model: Path, ids: list[int], max_new: int) -> None:
    """The channel does not change greedy output, and every DATA carries its token's tail."""
    plain = Serve(binary, model)
    try:
        without = plain.submit(token_prompt(ids), max_new)
    finally:
        plain.close()
    channel = Serve(binary, model)
    try:
        with_lp = channel.submit(token_prompt(ids), max_new, logprobs=2)
    finally:
        channel.close()
    if without.text != with_lp.text:
        raise AssertionError(f"logprobs changed the output:\n  plain: {without.text!r}\n  channel: {with_lp.text!r}")
    if not with_lp.data:
        raise AssertionError("no DATA frames with the channel open")
    for piece, tail in with_lp.data:
        if tail is None:
            raise AssertionError(f"DATA {piece!r} carries no logprob tail")
        token = parse_tokens(piece.decode("utf-8", "replace"))
        if len(token) != 1:
            raise AssertionError(f"DATA frame is not one token: {piece!r}")
        if len(tail["top"]) != 2:
            raise AssertionError(f"asked top-2, got {len(tail['top'])}")
        # greedy: the emitted token IS the best one, so its lp is the top entry
        if best_of(tail) != token[0] or abs(tail["top"][token[0]] - tail["lp"]) > 1e-5:
            raise AssertionError(f"DATA {token}: tail {tail} does not name it as the best token")
    print(f"PASS generation: {len(with_lp.data)} tokens unchanged, each DATA frame with its tail")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    arguments = parser.parse_args()
    binary = arguments.binary.resolve()
    reference = json.loads((arguments.fixture / "ref.json").read_text("utf-8"))
    case = reference["cases"]["short"]
    ids = list(case["prompt_ids"])
    greedy = list(case["greedy_new_ids"])
    # two options: the token greedy decoding would pick, and one it would not
    a = greedy[0]
    b = next(t for t in ids if t != a)
    check_read_only(binary, arguments.fixture, ids)
    check_echo_is_the_head(binary, arguments.fixture, ids)
    check_pin(binary, arguments.fixture, ids, a, b)
    check_generation_with_tail(binary, arguments.fixture, ids, 4)
    print("PASS DeepSeek V4 numeric channel: all checks completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
