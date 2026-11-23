#!/usr/bin/env python3
"""A short request between two turns keeps the conversation (park.h) on the tiny K3 fixture.

A client's title or summary request between two turns of a conversation took the only
state there is: the 69 KDA layers' recurrence cannot be rewound, so the conversation's next
turn read everything again. The conversation is parked for it (every KDA state whole, the
MLA rows the request may write) and the next request decides: back, kept parked or dropped.

What is asserted, all deterministic:
  - the short request parks the conversation and the next turn brings it back;
  - the next turn reuses the whole of turn 1's prompt instead of reading it again;
  - both answer byte for byte as a cold engine does;
  - COLI_PARK=0: the next turn reads the conversation again, and still answers the same.
  - #2703: a new conversation shorter than the last parks it and keeps its own state: its
    second turn reuses its first, the old conversation is dropped, the answer is a cold engine's;
  - a second short request that fits the rows kept apart leaves the conversation parked, and the
    next turn still brings it back whole.

The derived fixture (config, weights, tests/tok_kimi_tiny.json) as test_kimi_k3_ckpt.py builds it.
"""

import argparse
import re
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_kimi_k3_ckpt import Serve  # noqa: E402

OPENING = ("Alpha beta gamma delta epsilon zeta. Eta theta iota kappa lambda mu.\n"
           "Nu xi omicron pi rho sigma tau upsilon.\n")
SIDE = "Question about apples?\n"
SIDE2 = "Question about apple?\n"   # a second short request: its start is the first's, not the conversation's
SECOND = OPENING + "Question about bridges?\n"
NEW = "Question about pears?\n"     # a new conversation, shorter than the last (#2703)
NEW2 = NEW + "Question about bridges?\n"
MAX_NEW = 6
ENV = {"COLI_PARK_MIN": "8", "COLI_PREFIX_LOG": "1"}


def session(binary, model, prompts, extra, stderr_path):
    serve = Serve(binary, model, dict(ENV, **extra), stderr_path)
    try:
        return [serve.submit(p, n) for p, n in prompts], Path(stderr_path).read_text(errors="replace")
    finally:
        serve.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--binary", required=True)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--tokenizer", default=str(Path(__file__).parent / "tok_kimi_tiny.json"))
    args = parser.parse_args()

    base = Path(args.fixture)
    work = Path(tempfile.mkdtemp(prefix="kimi-k3-park-"))
    try:
        model = work / "fixture"
        model.mkdir()
        shutil.copy(base / "config.json", model / "config.json")
        shutil.copy(base / "model.safetensors", model / "model.safetensors")
        shutil.copy(args.tokenizer, model / "tokenizer.json")
        turns = [(OPENING, 1), (SIDE, MAX_NEW), (SECOND, MAX_NEW)]

        (opening, side, second), log = session(args.binary, model, turns, {}, work / "park.stderr")
        if "conversation parked for a" not in log or "conversation is back" not in log:
            raise AssertionError("the short request did not park the conversation:\n" + log)
        reused = [int(n) for n in re.findall(r"\[PREFIX\] reusing (\d+) of", log)]
        if not reused or reused[-1] < opening[1]:
            raise AssertionError(f"the next turn read the parked conversation again (turn 1 held "
                                 f"{opening[1]} tokens):\n" + log)
        print(f"ok park: the next turn reused {reused[-1]} tokens, turn 1's {opening[1]}")

        for name, (prompt, n), got in (("short request", turns[1], side), ("next turn", turns[2], second)):
            (cold,), _ = session(args.binary, model, [(prompt, n)], {}, work / "cold.stderr")
            if got[0] != cold[0]:
                raise AssertionError(f"the {name} answered {got[0]!r} beside a parked conversation, "
                                     f"{cold[0]!r} on a cold engine")
        print("ok identity: the short request and the next turn answer as a cold engine")

        (_, _, off), log_off = session(args.binary, model, turns, {"COLI_PARK": "0"}, work / "off.stderr")
        if "parked" in log_off:
            raise AssertionError("COLI_PARK=0 still parked:\n" + log_off)
        reused_off = [int(n) for n in re.findall(r"\[PREFIX\] reusing (\d+) of", log_off)]
        if reused_off and reused_off[-1] >= opening[1]:
            raise AssertionError("COLI_PARK=0 kept the conversation anyway:\n" + log_off)
        if off[0] != second[0]:
            raise AssertionError(f"parking changed the next turn's answer: {second[0]!r} vs {off[0]!r}")
        print("ok off: COLI_PARK=0 reads the conversation again and answers the same")

        # #2703: a new conversation shorter than the last parks it, and its own second turn must
        # find its own state, not the old conversation brought back after its first turn
        # (one token for the new conversation's first turn, as for the opening: the state then holds its
        # prompt, which its second turn begins with; a recurrence cannot leave an answer it held)
        turns = [(OPENING, 1), (NEW, 1), (NEW2, MAX_NEW)]
        (_, new1, new2), log = session(args.binary, model, turns, {}, work / "new.stderr")
        if "conversation parked for a" not in log:
            raise AssertionError("the shorter conversation did not park the old one:\n" + log)
        if "conversation is back" in log:
            raise AssertionError("the old conversation came back after the new one's turn:\n" + log)
        reused = [int(n) for n in re.findall(r"\[PREFIX\] reusing (\d+) of", log)]
        if not reused or reused[-1] < new1[1]:
            raise AssertionError(f"the new conversation's second turn read its first turn ({new1[1]} "
                                 f"tokens) again:\n" + log)
        (cold,), _ = session(args.binary, model, [(NEW2, MAX_NEW)], {}, work / "cold.stderr")
        if new2[0] != cold[0]:
            raise AssertionError(f"the new conversation's second turn answered {new2[0]!r}, "
                                 f"{cold[0]!r} on a cold engine")
        print(f"ok new conversation: its second turn reused {reused[-1]} tokens, the old one dropped")

        # two short requests between two turns: the second fits the rows kept apart and leaves the
        # conversation parked, starting over on those rows alone; the next turn brings it back whole
        turns = [(OPENING, 1), (SIDE, MAX_NEW), (SIDE2, MAX_NEW), (SECOND, MAX_NEW)]
        (opening, _, side2, second2), log = session(args.binary, model, turns, {}, work / "two.stderr")
        if "conversation stays parked" not in log or log.count("conversation is back") != 1:
            raise AssertionError("the second short request did not keep the conversation parked:\n" + log)
        reused = [int(n) for n in re.findall(r"\[PREFIX\] reusing (\d+) of", log)]
        if not reused or reused[-1] < opening[1]:
            raise AssertionError("the next turn read the parked conversation again:\n" + log)
        for name, (prompt, n), got in (("second short request", turns[2], side2), ("next turn", turns[3], second2)):
            (cold,), _ = session(args.binary, model, [(prompt, n)], {}, work / "cold.stderr")
            if got[0] != cold[0]:
                raise AssertionError(f"the {name} answered {got[0]!r} after two short requests, "
                                     f"{cold[0]!r} on a cold engine")
        print(f"ok two short requests: the conversation stayed parked, the next turn reused {reused[-1]} tokens")
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
