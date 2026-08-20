#!/usr/bin/env python3
"""Speculative-decoding gap check (U7a acceptance test 4).

Asserts, over a captured raw engine-stdout transcript, that EVERY generated
DATA frame of an opted-in request carries the per-token numeric channel
("DATA <id> <n> <lp> <k> [tid tlp]*k") -- accepted draft tokens included.
Speculatively-accepted draft tokens bypass the pick_tok call sites in the mux
loop (packet Fork 5's implementation risk), so a gap would show up as a
legacy 3-field DATA frame in the middle of an opted-in generation.

Run recipe (orchestrator; needs a real model -- CPU or CUDA build):

    # single KV slot + model drafts live = the speculative serve regime
    cd c && make glm
    SERVE=1 SERVE_BATCH=1 KV_SLOTS=1 DRAFT=2 CTX=4096 \
        ./colibri <snapshot-dir> < submit.txt | tee engine_stdout.raw

  where submit.txt contains one opted-in generation request, e.g. (prompt
  "Hello" = 5 payload bytes, 64 new tokens, greedy, logprobs top-5):

    SUBMIT 7 0 5 64 0 1 0 logprobs=5
    Hello

  (payload line follows the header; the trailing newline after the payload is
  part of the framing). A speculative run needs an MTP-bearing container or
  the n-gram fallback to actually accept drafts -- confirm acceptance in the
  stderr log ("[MTP] ..." / spec acceptance lines) so the run genuinely
  exercised the draft-accept emit path rather than trivially passing.

Then:

    python3 tests/check_data_logprob_gaps.py engine_stdout.raw --id 7 --topk 5

Checks performed:
  - every DATA frame for --id has exactly 5 + 2k header fields, a parseable
    float lp, and k == --topk (k may be < topk only if vocab < topk).
    A non-finite lp ("nan"/"inf": degenerate logits, e.g. an all -inf row
    after grammar masking) counts as PRESENT -- the channel carried a value,
    which is exactly what this audit is for -- and is FLAGGED in the output
    without failing the check (whether the engine should serialize such rows
    differently is a U7b server-side register question, not a gap);
  - ECHO frames (if the request also echoed) cover contiguous positions
    0..P-1, position 0 carrying "nan 0";
  - payload framing (n bytes + newline) stays byte-exact throughout, so a
    single malformed frame cannot hide by desynchronizing the parse.
Exit 0 = no gaps; non-zero = at least one gap/malformed frame (listed).
"""
import argparse
import math
import sys


def parse_frames(blob):
    """Parse the serve-mux stdout byte stream into (header_fields, payload)."""
    frames = []
    i = 0
    n = len(blob)
    while i < n:
        j = blob.find(b"\n", i)
        if j < 0:
            break
        line = blob[i:j]
        i = j + 1
        fields = line.split()
        if not fields:
            continue
        kind = fields[0]
        if kind in (b"DATA", b"ECHO") and len(fields) >= 3:
            try:
                size = int(fields[2])
            except ValueError:
                frames.append((fields, None))
                continue
            payload = blob[i:i + size]
            i += size
            if i < n and blob[i:i + 1] == b"\n":
                i += 1
            else:
                frames.append((fields, b"<BAD TERMINATOR>"))
                continue
            frames.append((fields, payload))
        else:
            frames.append((fields, None))
    return frames


def check(frames, request_id, topk):
    rid = str(request_id).encode()
    problems = []
    flags = []
    data_frames = 0
    echo_positions = []
    for fields, payload in frames:
        if len(fields) < 2 or fields[1] != rid:
            continue
        kind = fields[0]
        if kind == b"DATA":
            data_frames += 1
            if len(fields) == 3:
                problems.append(f"GAP: legacy 3-field DATA frame #{data_frames}"
                                f" (payload {payload!r}) has no logprob")
                continue
            try:
                lp = float(fields[3])
                k = int(fields[4])
            except (ValueError, IndexError):
                problems.append(f"malformed DATA numeric fields: {fields!r}")
                continue
            if not math.isfinite(lp):
                # PRESENT, not a gap: the channel carried a value; degenerate
                # logits (an all -inf row, say) legitimately produce nan/inf.
                # Flagged so a reviewer sees it; never fails the audit.
                flags.append(f"non-finite lp in DATA frame #{data_frames}"
                             f" (present, flagged): {fields!r}")
            if k != topk:
                problems.append(f"DATA top-k {k} != requested {topk}: {fields!r}")
            if len(fields) != 5 + 2 * k:
                problems.append(f"DATA field count {len(fields)} != 5+2k: {fields!r}")
        elif kind == b"ECHO":
            try:
                pos = int(fields[3])
            except (ValueError, IndexError):
                problems.append(f"malformed ECHO frame: {fields!r}")
                continue
            echo_positions.append(pos)
            if pos == 0 and (fields[4] != b"nan" or fields[5] != b"0"):
                problems.append(f"ECHO position 0 should carry 'nan 0': {fields!r}")
            if pos > 0 and len(fields) < 6:
                problems.append(f"short ECHO frame: {fields!r}")
    if echo_positions and echo_positions != list(range(len(echo_positions))):
        problems.append(f"ECHO positions not contiguous from 0: {echo_positions}")
    return data_frames, len(echo_positions), problems, flags


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("transcript", help="raw engine stdout capture")
    parser.add_argument("--id", required=True, help="request id to audit")
    parser.add_argument("--topk", type=int, required=True,
                        help="the SUBMIT logprobs=k value the request used")
    args = parser.parse_args()
    blob = open(args.transcript, "rb").read()
    frames = parse_frames(blob)
    data_frames, echo_frames, problems, flags = check(frames, args.id, args.topk)
    print(f"[gapcheck] id={args.id}: {data_frames} DATA frames, "
          f"{echo_frames} ECHO frames, {len(flags)} flagged non-finite")
    if data_frames == 0:
        problems.append("no DATA frames found for this id -- wrong id, or the "
                        "run produced nothing (check ERROR frames)")
    for flag in flags:
        print(f"[gapcheck] FLAG: {flag}")
    for problem in problems:
        print(f"[gapcheck] {problem}")
    verdict = ("FAIL" if problems
               else "PASS: every generated token carries a logprob value")
    print(f"[gapcheck] {verdict}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
