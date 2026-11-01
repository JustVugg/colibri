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
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def token_prompt(ids: list[int]) -> str:
    return "".join(f"<t{token:03d}>" for token in ids)


def parse_tokens(text: str) -> list[int]:
    """The tiny fixture's vocabulary is <tNNN>, so a reply decodes back to ids."""
    return [int(match) for match in re.findall(r"<t(\d+)>", text)]


class Serve:
    """One persistent `SERVE=1` engine process."""

    def __init__(self, binary: Path, model: Path, ctx: str = "128",

                 extra: dict[str, str] | None = None) -> None:
        env = dict(os.environ, SERVE="1", SNAP=str(model), CTX=ctx,
                   V4_PREFIX_LOG="1", **(extra or {}))
        self.process = subprocess.Popen(
            [str(binary)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, env=env,
        )
        self.counter = 0
        self.stderr_text = ""

    def submit(self, prompt: str, max_tokens: int,
               prefix_bytes: int = 0) -> tuple[str, int]:
        """Returns the generated text and the DONE frame's reuse count.
        prefix_bytes is the optional gateway hint (the 8th SUBMIT field: the
        byte offset where the rendered system turn ends)."""
        self.counter += 1
        request_id = f"r{self.counter}"
        payload = prompt.encode("utf-8")
        hint = f" 0 {prefix_bytes}" if prefix_bytes else ""
        header = (f"SUBMIT {request_id} 0 {len(payload)} {max_tokens} "
                  f"0.0 1.0{hint}\n").encode("ascii")
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
        finally:
            assert self.process.stderr is not None
            try:
                self.stderr_text = self.process.stderr.read().decode(
                    "utf-8", "replace")
            except Exception:                      # pragma: no cover
                self.stderr_text = ""


def check_growing_conversation(binary: Path, model: Path,
                               case: dict[str, object],
                               extra_env: dict[str, str] | None = None) -> None:
    """Turn 2 extends turn 1: the state is reused, the output does not change."""
    first_ids = list(case["prompt_ids"])            # type: ignore[arg-type]
    max_new = 4

    warm = Serve(binary, model, extra_env=extra_env)
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

    cold = Serve(binary, model, extra_env=extra_env)
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



def check_short_request_parks(binary: Path, model: Path,
                              case: dict[str, object]) -> None:
    """A short request between two turns keeps the conversation (park.h).

    A client's title or summary request between two turns shares nothing with the
    conversation and took the only state there is: the next turn re-prefilled everything.
    The conversation is parked for it and comes back: the next turn reuses the whole of
    turn 1's prompt, and both answer as a cold engine does."""
    base = list(case["prompt_ids"])                 # type: ignore[arg-type]
    first_ids = (base * (48 // len(base) + 1))[:48]
    side_ids = [token + 1 for token in base[:4]]
    second_ids = first_ids + base[:2]
    max_new = 4

    def run(extra: dict[str, str]) -> tuple[list[tuple[str, int]], str]:
        warm = Serve(binary, model, extra=dict(COLI_PARK_MIN="16", COLI_PREFIX_LOG="1", **extra))
        try:
            out = [warm.submit(token_prompt(first_ids), 1),
                   warm.submit(token_prompt(side_ids), max_new),
                   warm.submit(token_prompt(second_ids), max_new)]
        finally:
            warm.close()
        assert warm.process.stderr is not None
        return out, warm.process.stderr.read().decode("utf-8", "replace")

    (_, (side_text, _), (second_text, second_reuse)), log = run({})
    if "conversation parked for a" not in log or "conversation is back" not in log:
        raise AssertionError("the short request did not park the conversation:\n" + log)
    if second_reuse < len(first_ids):
        raise AssertionError(f"the next turn reused {second_reuse} tokens, not the "
                             f"{len(first_ids)} of the parked conversation:\n" + log)
    for name, ids, got in (("short request", side_ids, side_text), ("next turn", second_ids, second_text)):
        cold = Serve(binary, model)
        try:
            cold_text, _ = cold.submit(token_prompt(ids), max_new)
        finally:
            cold.close()
        if got != cold_text:
            raise AssertionError(f"the {name} answered {got!r} beside a parked conversation, "
                                 f"{cold_text!r} on a cold engine")
    (_, _, (off_text, off_reuse)), _ = run({"COLI_PARK": "0"})
    if off_reuse >= len(first_ids) or off_text != second_text:
        raise AssertionError(f"COLI_PARK=0: reuse {off_reuse}, answer {off_text!r} (parked: {second_text!r})")
    print(f"PASS prefix park: the next turn reused {second_reuse} tokens after a short "
          f"request, answers identical to a cold engine")

def check_cold_tier_disk(binary: Path, fixture: Path) -> None:
    """The disk catalog across restarts and interleaved lineages (P1.1).

    Runs on a COPY of the tiny fixture (the checkpoint directory lives inside
    the model dir, and no test may write into a tracked path) with the capture
    floor lowered to 64 (the fixture's context is 128 tokens) and prompt-end
    persistence on (V4_PREFIX_CKPT_DISK=2; the default 1 persists prefix
    captures only -- flipping that default is a separate gated change):
      - cold restart: agent A captures the shared 64-token system prefix via
        the gateway hint; a FRESH process on the same model dir restores it
        (stderr: disk load + hit, DONE reuse=64) instead of re-prefilling;
      - interleaving A/B/A/B: each turn restores the other lineage's stored
        boundary (live strict-prefix reuse can never fire there);
      - mid-fork: a prompt sharing only 32 tokens with every boundary has no
        stored strict prefix and re-prefills (reuse=0).
    """
    work = Path(tempfile.mkdtemp(prefix="v4_cold_tier_"))
    model = work / "tiny"
    shutil.copytree(fixture, model)
    env = {"V4_PREFIX_CKPT_DISK": "2", "V4_PREFIX_CKPT_MIN": "64"}
    try:
        system = [5 + (index * 11) % 97 for index in range(64)]
        tail_a = [5 + (index * 13) % 89 for index in range(20)]
        tail_b = [5 + (index * 17) % 83 for index in range(20)]
        prompt_a1 = system + tail_a
        prompt_b1 = system + tail_b
        hint = 64 * len("<t000>")   # the byte offset where the system turn ends

        warm = Serve(binary, model, extra_env=env)
        try:
            text_a1, reuse = warm.submit(token_prompt(prompt_a1), 4,
                                         prefix_bytes=hint)
            if reuse != 0:
                raise AssertionError(f"first turn reused {reuse} tokens")
            reply_a = parse_tokens(text_a1)
            prompt_b1_text, reuse_b1 = warm.submit(token_prompt(prompt_b1), 4)
            if reuse_b1 != 64:
                raise AssertionError(
                    f"same-process lineage switch restored {reuse_b1}, want 64")
            # A2 extends A1 (prompt + reply minus the last token + new text):
            # the live state holds B's lineage, so only the stored A1
            # prompt-end boundary can serve this turn.
            prompt_a2 = prompt_a1 + (reply_a[:-1] if len(reply_a) > 1 else [])
            prompt_a2 += [40 + index for index in range(4)]
            _, reuse_a2 = warm.submit(token_prompt(prompt_a2), 4)
            if reuse_a2 < len(prompt_a1):
                raise AssertionError(
                    f"A2 restored only {reuse_a2} tokens, want the stored "
                    f"{len(prompt_a1)}-token prompt-end boundary")
            prompt_b2 = prompt_b1 + [60 + index for index in range(4)]
            _, reuse_b2 = warm.submit(token_prompt(prompt_b2), 4)
            if reuse_b2 < len(prompt_b1):
                raise AssertionError(
                    f"B2 restored only {reuse_b2} tokens, want the stored "
                    f"{len(prompt_b1)}-token prompt-end boundary")
        finally:
            warm.close()
        if "v4_ckpt store prefix=64" not in warm.stderr_text:
            raise AssertionError(
                "the gateway hint never stored the system prefix:\n"
                + warm.stderr_text)

        # Cold restart: a fresh process scans the model dir and restores the
        # stored system prefix instead of re-prefilling it.
        fresh = Serve(binary, model, extra_env=env)
        try:
            _, reuse = fresh.submit(token_prompt(prompt_b1), 4)
        finally:
            fresh.close()
        if reuse != 64:
            raise AssertionError(
                f"cold restart restored {reuse} tokens, want 64")
        for line in ("v4_ckpt disk load prefix=64", "v4_ckpt hit prefix=64"):
            if line not in fresh.stderr_text:
                raise AssertionError(
                    f"missing {line!r} on the cold restart:\n"
                    + fresh.stderr_text)

        # Mid-fork: shares only the first 32 tokens with every stored
        # boundary -- no stored strict prefix exists, so it re-prefills.
        forked = Serve(binary, model, extra_env=env)
        try:
            _, reuse = forked.submit(
                token_prompt(system[:32] + [70 + index for index in range(20)]),
                4)
        finally:
            forked.close()
        if reuse != 0:
            raise AssertionError(
                f"mid-fork prompt restored {reuse} tokens; a boundary inside "
                "an entry cannot be rewound to")
        print("PASS cold tier: restart restore, interleaved A/B/A/B, "
              "mid-fork re-prefill")
    finally:
        shutil.rmtree(work, ignore_errors=True)


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

    check_short_request_parks(arguments.binary, arguments.fixture, case)
    check_cold_tier_disk(arguments.binary, arguments.fixture)
    # The same live-reuse property against the V4_PREFIX_CKPT_DISK=0 memory
    # backend: the catalog is the default, the slot pool must still work.
    check_growing_conversation(arguments.binary, arguments.fixture, case,
                               extra_env={"V4_PREFIX_CKPT_DISK": "0"})
    print("PASS DeepSeek V4 KV prefix reuse: all checks completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
