#!/usr/bin/env python3
"""KV prefix reuse on the DeepSeek V4 serve path.

The property under test is not "it is faster" -- it is that reusing the
attention state produces the bytes a cold prefill would have produced. So each
case runs the same second turn twice: once as the continuation of a warm
session, once against a freshly started engine, and requires the two to agree
token for token.

Speaking the SUBMIT/DATA/DONE protocol directly rather than through
openai_server.Engine keeps the test on the engine's own contract, including the
trailing reuse field of the DONE frame.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


def token_prompt(ids: list[int]) -> str:
    return "".join(f"<t{token:03d}>" for token in ids)


def parse_tokens(text: str) -> list[int]:
    """The tiny fixture's vocabulary is <tNNN>, so a reply decodes back to ids."""
    return [int(match) for match in re.findall(r"<t(\d+)>", text)]


class Serve:
    """One persistent `SERVE=1` engine process."""

    def __init__(self, binary: Path, model: Path, ctx: str = "128") -> None:
        env = dict(os.environ, SERVE="1", SNAP=str(model), CTX=ctx,
                   V4_PREFIX_LOG="1")
        self.process = subprocess.Popen(
            [str(binary)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, env=env,
        )
        self.counter = 0

    def submit(self, prompt: str, max_tokens: int) -> tuple[str, int]:
        """Returns the generated text and the DONE frame's reuse count."""
        self.counter += 1
        request_id = f"r{self.counter}"
        payload = prompt.encode("utf-8")
        header = (f"SUBMIT {request_id} 0 {len(payload)} {max_tokens} "
                  f"0.0 1.0 0\n").encode("ascii")
        assert self.process.stdin is not None
        self.process.stdin.write(header + payload + b"\n")
        self.process.stdin.flush()

        pieces: list[bytes] = []
        reuse = -1
        stream = self.process.stdout
        assert stream is not None
        while True:
            line = stream.readline()
            if not line:
                raise AssertionError(
                    f"engine closed stdout during {request_id}; "
                    f"stderr:\n{self._drain_stderr()}"
                )
            fields = line.decode("utf-8", "replace").split()
            if not fields:
                continue
            if fields[0] == "ERROR":
                raise AssertionError(f"engine ERROR: {line!r}")
            if fields[0] == "DATA" and len(fields) == 3:
                size = int(fields[2])
                pieces.append(stream.read(size))
                stream.read(1)                     # trailing newline
            elif fields[0] == "DONE":
                # DONE <id> STAT <completion> <tok/s> <hit%> <rss> <prompt> <cap> [<reuse>]
                if len(fields) >= 10:
                    reuse = int(fields[9])
                break
        return b"".join(pieces).decode("utf-8", "replace"), reuse

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


def check_growing_conversation(binary: Path, model: Path,
                               case: dict[str, object]) -> None:
    """Turn 2 extends turn 1: the state is reused, the output does not change."""
    first_ids = list(case["prompt_ids"])            # type: ignore[arg-type]
    max_new = 4

    warm = Serve(binary, model)
    try:
        first_text, first_reuse = warm.submit(token_prompt(first_ids), max_new)
        if first_reuse != 0:
            raise AssertionError(
                f"first turn of a fresh session reused {first_reuse} tokens; "
                "there was nothing to reuse"
            )
        # What the engine feeds is the prompt plus every generated token except
        # the last, which is emitted but never fed back. Turn 2 has to be a
        # STRICT extension of that, which is what a chat produces: the previous
        # prompt, the reply, then the new user text.
        reply = parse_tokens(first_text)
        if len(reply) < 2:
            raise AssertionError(f"first turn produced too little: {first_text!r}")
        fed = first_ids + reply[:-1]
        second_ids = fed + [first_ids[0]]
        second_text, second_reuse = warm.submit(token_prompt(second_ids), max_new)
    finally:
        warm.close()

    cold = Serve(binary, model)
    try:
        cold_text, cold_reuse = cold.submit(token_prompt(second_ids), max_new)
    finally:
        cold.close()

    if cold_reuse != 0:
        raise AssertionError(f"cold engine reported reuse={cold_reuse}")
    if second_text != cold_text:
        raise AssertionError(
            "warm continuation diverged from a cold prefill of the same prompt:\n"
            f"  warm: {second_text!r}\n  cold: {cold_text!r}"
        )
    if second_reuse <= 0:
        raise AssertionError(
            "prefix reuse never fired on a prompt that extends the previous "
            f"turn (reuse={second_reuse}); the optimisation is inert"
        )
    if second_reuse < len(first_ids):
        raise AssertionError(
            f"reused only {second_reuse} tokens, expected at least the "
            f"{len(first_ids)} prompt tokens of the previous turn"
        )
    print(f"PASS prefix reuse: {second_reuse} tokens reused, "
          f"output identical to a cold prefill")


def check_divergent_prompt_resets(binary: Path, model: Path,
                                  case: dict[str, object]) -> None:
    """A prompt that is not an extension must fall back to a full prefill."""
    first_ids = list(case["prompt_ids"])            # type: ignore[arg-type]
    other_ids = [token + 1 for token in first_ids]
    max_new = 4

    warm = Serve(binary, model)
    try:
        warm.submit(token_prompt(first_ids), max_new)
        divergent_text, reuse = warm.submit(token_prompt(other_ids), max_new)
    finally:
        warm.close()

    cold = Serve(binary, model)
    try:
        cold_text, _ = cold.submit(token_prompt(other_ids), max_new)
    finally:
        cold.close()

    if reuse != 0:
        raise AssertionError(
            f"reused {reuse} tokens from a prompt that shares no full prefix"
        )
    if divergent_text != cold_text:
        raise AssertionError(
            "divergent prompt did not reset the state:\n"
            f"  warm: {divergent_text!r}\n  cold: {cold_text!r}"
        )
    print("PASS prefix reset: divergent prompt re-prefills and matches cold")


def check_repeated_prompt(binary: Path, model: Path,
                          case: dict[str, object]) -> None:
    """An identical prompt is not a strict extension: it must re-prefill."""
    ids = list(case["prompt_ids"])                  # type: ignore[arg-type]
    max_new = 4

    warm = Serve(binary, model)
    try:
        first_text, _ = warm.submit(token_prompt(ids), max_new)
        second_text, reuse = warm.submit(token_prompt(ids), max_new)
    finally:
        warm.close()

    if reuse != 0:
        raise AssertionError(f"identical prompt reported reuse={reuse}")
    if first_text != second_text:
        raise AssertionError(
            f"repeated prompt changed answer: {first_text!r} vs {second_text!r}"
        )
    print("PASS prefix repeat: identical prompt re-prefills, answer unchanged")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--fixture", required=True, type=Path)
    arguments = parser.parse_args()

    reference = json.loads((arguments.fixture / "ref.json").read_text("utf-8"))
    case = reference["cases"]["short"]

    check_growing_conversation(arguments.binary, arguments.fixture, case)
    check_repeated_prompt(arguments.binary, arguments.fixture, case)
    check_divergent_prompt_resets(arguments.binary, arguments.fixture, case)
    print("PASS DeepSeek V4 KV prefix reuse: all checks completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
