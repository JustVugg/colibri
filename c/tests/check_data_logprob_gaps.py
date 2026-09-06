#!/usr/bin/env python3
"""Speculative-decoding gap check (U7a acceptance test 4).

Asserts, over a captured raw engine-stdout transcript, that EVERY generated
DATA frame of an opted-in request carries the per-token numeric channel
("DATA <id> <n> <lp> <k> [tid tlp]*k") -- accepted draft tokens included.
On today's engine, an accepted speculative-draft token is emitted through
the SAME mux_data() call as any other generated token (mux_spec_emit ->
mux_data, passing the same logits row and requested top-k the ordinary
emit sites use), so the gap this check hunts -- a legacy 3-field DATA
frame appearing mid-generation because a draft-accept path bypassed the
numeric-channel emit -- is NOT producible by the engine as shipped. This
check is retained as REGRESSION COVERAGE for that invariant (a future mux
change that adds a new emit call site without the tail would reintroduce
exactly this defect class), not as a currently-live defect hunt.

Run recipe (orchestrator; needs a real model -- CPU or explicit-CUDA build):

    # single KV slot + model drafts live = the speculative serve regime
    cd c && make glm
    SERVE=1 SERVE_BATCH=1 KV_SLOTS=1 DRAFT=2 CTX=4096 \
        ./colibri <snapshot-dir> < submit.raw | tee engine_stdout.raw

  where submit.raw contains one opted-in generation request, e.g. (prompt
  "Hello" = 5 payload bytes, 64 new tokens, greedy, logprobs top-5):

    SUBMIT 7 0 5 64 0 1 0 logprobs=5
    Hello

  (payload line follows the header; the trailing newline after the payload is
  part of the framing). A speculative run needs an MTP-bearing container or
  the n-gram fallback to actually accept drafts -- confirm acceptance in the
  stderr log ("[MTP] ..." / spec acceptance lines) so the run genuinely
  exercised the draft-accept emit path rather than trivially passing.

A raw capture taken this way is a plain pipeline: it is not itself hash-bound
to the binary, container, input, and environment that produced it, and
NOTHING in this module binds it either -- the preamble gate below checks
only that the BANNER/LOADED text is a well-formed, self-consistent record of
what the engine printed about itself; it does not verify that text against
an independently computed binary or container digest. Whichever capture step
feeds this checker, prefer one that additionally binds and retains that
provenance (distinct raw stdout/stderr, the direct engine exit status, and a
hash over the binary/container/input/environment/protocol payloads) over a
bare pipeline or a lone DATA/HIT record, which is not decisive evidence on
its own.

Then, against the resulting transcript, the numeric-channel check runs:

    python3 tests/check_data_logprob_gaps.py engine_stdout.raw \
        --id 7 --topk 5 --vocab 154880

Checks performed:
  - the transcript's capture shape (full-process / ready-suffix / request-
    only) is named by capture_mode() and enforced explicitly: the leading
    line must be a recognized global preamble record or a legitimate
    request-frame kind, and a transcript with any global record present
    but not correctly led (a "invalid" capture per capture_mode()) fails
    loud -- no transcript reaches the rest of the checks unclassified;
  - the startup BANNER/LOADED preamble, where present, parses exactly
    (the engine_evidence grammar); an unparsed preamble is a named
    failure quoting the offending line, never a silent pass;
  - exactly one ACCEPT and one DONE exist for --id, DONE's emitted count
    equals the positive DATA count, and no targeted ERROR or post-DONE
    frame exists;
  - every DATA/ECHO numeric tail has exactly the advertised fields and
    min(--topk,--vocab) unique token ids in range. Each numeric token
    (target logprob and every top-k logprob) must be a finite,
    non-positive value spelled as the fixed six-decimal form dev's
    engine prints today ("fixed6", e.g. "-0.300000"), the %.17g form an
    earlier engine build printed ("c17g", e.g. "-2.7000000000000002"), or
    an exact nan/inf/-inf spelling -- whichever form parses. A token that
    happens to satisfy BOTH spellings exactly (e.g. '%.17g' % -1.234567
    == '-1.234567', itself also an exact six-decimal spelling) is
    "ambiguous": there is no way to tell, from the token alone, which
    engine format produced it, so no per-frame or per-transcript
    "consistent format" rule is enforced -- one was tried and had to be
    dropped (see the module history) because it produced false rejections
    on real %.17g transcripts purely from this ambiguity. The observed
    form counts (fixed6-only / c17g-only / ambiguous / special) are
    reported in the summary line as INFORMATION ONLY; they never affect
    the verdict;
  - ACCEPT's canonical prompt length P has exactly P ECHO frames at
    positions 0..P-1, position 0 carrying "nan 0";
  - payload framing (n bytes + newline) stays byte-exact throughout, so a
    single malformed frame cannot hide by desynchronizing the parse; every
    reported problem cites the byte offset of the offending frame's own
    header line (not the frame that follows it).
Exit 0 = no gaps; non-zero = at least one gap/malformed frame (listed).
"""
import argparse
import collections
import math
import os
import re
import sys


_TOOLS_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "tools")
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)
from engine_evidence import PreambleError, parse_engine_preamble


_INT32_MAX = 2**31 - 1
_UINT64_MAX = 2**64 - 1
_UINT_RE = re.compile(rb"(?:0|[1-9][0-9]*)")
_C17G_RE = re.compile(
    rb"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?"
    rb"(?:e[+-](?:0[0-9]|[1-9][0-9]{1,2}))?")
_FIXED6_RE = re.compile(rb"-?(?:0|[1-9][0-9]*)\.[0-9]{6}")
_SPECIAL_TOKENS = {b"nan": math.nan, b"inf": math.inf, b"-inf": -math.inf}
_READY = b"\x01\x01READY\x01\x01"
_GLOBAL_KINDS = frozenset((
    b"BANNER", b"LOADED", b"READY", b"STAT", b"HWINFO", b"TIERS",
    b"EMAP", b"HITS", b"PROF",
))
_TARGETED_KINDS = frozenset((
    b"ACCEPT", b"DATA", b"ECHO", b"DONE", b"ERROR",
))
_LOWER_HEX_RE = re.compile(rb"[0-9a-f]*")


def _uint(token, label, maximum=_INT32_MAX):
    if not _UINT_RE.fullmatch(token):
        raise ValueError(f"noncanonical {label}: {token!r}")
    value = int(token.decode("ascii"))
    if value > maximum:
        raise ValueError(f"{label} exceeds {maximum}: {token!r}")
    return value


def _c17g(token, label):
    if not _C17G_RE.fullmatch(token):
        raise ValueError(f"noncanonical {label} %.17g token: {token!r}")
    value = float(token.decode("ascii"))
    if not math.isfinite(value) or format(value, ".17g").encode("ascii") != token:
        raise ValueError(f"not an exact finite {label} %.17g spelling: {token!r}")
    return value


def _fixed6(token, label):
    if not _FIXED6_RE.fullmatch(token):
        raise ValueError(f"noncanonical {label} %.6f token: {token!r}")
    value = float(token.decode("ascii"))
    if not math.isfinite(value) or format(value, ".6f").encode("ascii") != token:
        raise ValueError(f"not an exact finite {label} %.6f spelling: {token!r}")
    return value


def _numeric_value(token, label):
    """Classify and parse one engine numeric-tail token.

    Returns (value, form). "special" is an exact nan/inf/-inf spelling --
    libc's own %f/%g rendering of a non-finite double, which either wire
    format's snprintf call can emit identically. Otherwise the token is
    checked against BOTH finite grammars independently (never short-
    circuited): the fixed six-decimal form dev's engine prints today
    ("fixed6") and the %.17g form an earlier engine build printed
    ("c17g"). A token can satisfy both -- '%.17g' % -1.234567 ==
    '-1.234567', which is ALSO the exact six-decimal spelling of that
    same double -- and when it does, the form is "ambiguous": there is no
    way to tell, from the token alone, which engine format produced it.
    A token matching neither raises ValueError.

    An earlier version of this function tried fixed6 first and reported
    it whenever fixed6 matched, silently hiding the ambiguous case; that
    misclassified a large fraction of genuine %.17g tokens as fixed6 and
    fed a since-removed "consistent form per frame" check false mixed-
    form rejections on real transcripts. Classification is now purely
    informational (see check()/main()) precisely because it cannot be
    made unambiguous from the token alone.
    """
    if token in _SPECIAL_TOKENS:
        return _SPECIAL_TOKENS[token], "special"
    value = None
    is_fixed6 = is_c17g = False
    try:
        value = _fixed6(token, label)
        is_fixed6 = True
    except ValueError:
        pass
    try:
        value = _c17g(token, label)
        is_c17g = True
    except ValueError:
        pass
    if is_fixed6 and is_c17g:
        return value, "ambiguous"
    if is_fixed6:
        return value, "fixed6"
    if is_c17g:
        return value, "c17g"
    raise ValueError(
        f"{label} matches neither the fixed6 nor the c17g nor the "
        f"nan/inf spelling: {token!r}")


def _fixed_metric(token, places, label, lower=0.0, upper=None):
    pattern=rb"-?(?:0|[1-9][0-9]*)\."+rb"[0-9]{"+str(places).encode()+rb"}"
    if not re.fullmatch(pattern,token):
        raise ValueError(f"noncanonical {label}: {token!r}")
    value=float(token.decode("ascii"))
    if not math.isfinite(value) or value<lower or (upper is not None and value>upper):
        raise ValueError(f"{label} outside [{lower},{upper}]: {token!r}")
    return value


def _header_fields(line, byte_offset):
    problems=[]
    if not line:
        return [],[f"blank protocol header at byte {byte_offset}"]
    if (any(byte<0x20 or byte>0x7e for byte in line) or
            line.startswith(b" ") or line.endswith(b" ") or b"  " in line):
        problems.append(f"noncanonical ASCII/space header at byte {byte_offset}: {line!r}")
    return line.split(),problems


def _global_header(line, byte_offset):
    """Validate one exact production mux-global line.

    Return None when the line is not a recognized global.  Recognized globals
    return a synthetic one-field marker so check() owns their process-level
    lifecycle before any numeric token can collide with a request id.
    """
    if (line.startswith(b"== GLM C engine") or
            line.startswith(b"loaded in")):
        kind = b"BANNER" if line.startswith(b"==") else b"LOADED"
        problems = []
        try:
            text = line.decode("ascii")
            parsed = parse_engine_preamble(text)
            if parsed is None or parsed["kind"].encode("ascii") != kind:
                raise PreambleError("owned preamble kind mismatch")
        except (UnicodeDecodeError, PreambleError) as exc:
            problems.append(
                f"malformed {kind.decode()} preamble at byte {byte_offset}: "
                f"{exc}; {line!r}")
        return [kind], problems

    if line == _READY:
        return [b"READY"], []

    kind = line.split(b" ", 1)[0]
    if kind == b"READY":
        return [b"READY"], [
            f"malformed READY global at byte {byte_offset}: expected {_READY!r}; {line!r}"
        ]
    if kind not in _GLOBAL_KINDS - {b"READY"}:
        return None
    problems = []
    try:
        if kind == b"STAT":
            fields = line.split(b" ")
            if (len(fields) != 5 or fields[1:4] != [b"0", b"0.00", b"0.0"]):
                raise ValueError("expected exact startup STAT fields")
            _fixed_metric(fields[4], 2, "STAT RSS")
        elif kind == b"HWINFO":
            # The final CPU|GPU field is produced by two %s conversions.  It is
            # printable free text and may contain repeated spaces; the six
            # numeric/structural prefixes remain exact single-space fields.
            fields = line.split(b" ", 6)
            if len(fields) != 7:
                raise ValueError("expected six HWINFO prefixes and CPU|GPU tail")
            _uint(fields[1], "HWINFO core count")
            _fixed_metric(fields[2], 1, "HWINFO total RAM")
            _fixed_metric(fields[3], 1, "HWINFO available RAM")
            _uint(fields[4], "HWINFO GPU count")
            _fixed_metric(fields[5], 1, "HWINFO total VRAM")
            tail = fields[6]
            if (tail.count(b"|") != 1 or
                    any(byte < 0x20 or byte > 0x7e for byte in tail)):
                raise ValueError("HWINFO tail is not printable CPU|GPU text")
        elif kind == b"TIERS":
            fields = line.split(b" ")
            if len(fields) != 6:
                raise ValueError("expected five TIERS fields")
            for index, label in enumerate(("VRAM", "RAM", "disk"), 1):
                _uint(fields[index], f"TIERS {label} count")
            _fixed_metric(fields[4], 2, "TIERS VRAM GB")
            _fixed_metric(fields[5], 2, "TIERS RAM GB")
        elif kind in (b"EMAP", b"HITS"):
            fields = line.split(b" ")
            if len(fields) != 4:
                raise ValueError(f"expected three {kind.decode()} fields")
            rows = _uint(fields[1], f"{kind.decode()} row count")
            cols = _uint(fields[2], f"{kind.decode()} column count")
            if cols < 1 or (kind == b"HITS" and rows < 1):
                raise ValueError(
                    f"{kind.decode()} rows/columns outside producer domain")
            payload = fields[3]
            if not _LOWER_HEX_RE.fullmatch(payload):
                raise ValueError(f"{kind.decode()} payload is not lowercase hex")
            cells = rows * cols
            expected = cells * 2 if kind == b"EMAP" else ((cells + 7) // 8) * 2
            if len(payload) != expected:
                raise ValueError(
                    f"{kind.decode()} payload length {len(payload)} != {expected}")
            if kind == b"EMAP":
                for index in range(cells):
                    cell = int(payload[2 * index:2 * index + 2], 16)
                    tier, heat = cell >> 6, cell & 0x3f
                    if tier > 2:
                        raise ValueError(
                            f"EMAP cell {index} tier {tier} outside [0,2]")
                    if heat > 32:
                        raise ValueError(
                            f"EMAP cell {index} heat {heat} outside [0,32]")
            elif cells & 7:
                final = int(payload[-2:], 16)
                used_mask = (1 << (cells & 7)) - 1
                if final & ~used_mask:
                    raise ValueError("HITS final byte has nonzero padding bits")
        else:  # PROF
            fields = line.split(b" ")
            if len(fields) != 10:
                raise ValueError("expected nine PROF fields")
            _fixed_metric(fields[1], 3, "PROF wall seconds")
            _uint(fields[2], "PROF prompt count")
            _uint(fields[3], "PROF completion count")
            for index, label in enumerate(
                    ("disk", "wait", "matmul", "attention", "head"), 4):
                _fixed_metric(fields[index], 3, f"PROF {label} seconds")
            _uint(fields[9], "PROF forward count", _UINT64_MAX)
    except (ValueError, IndexError) as exc:
        problems.append(
            f"malformed {kind.decode()} global at byte {byte_offset}: {exc}; {line!r}")
    return [kind], problems


def parse_frames(blob):
    """Parse frames fail-closed, returning (frames, framing_problems).

    Each frame is (fields, payload, byte_offset), where byte_offset is the
    offset of THIS frame's own header line -- captured before the cursor
    advances past it, so every problem this module reports can cite the
    byte offset of the frame it is actually complaining about, not the
    frame that happens to follow it in the transcript.
    """
    frames = []
    problems = []
    i = 0
    n = len(blob)
    while i < n:
        line_start = i
        j = blob.find(b"\n", i)
        if j < 0:
            problems.append(f"truncated header at byte {line_start}")
            break
        line = blob[i:j]
        i = j + 1
        global_header = _global_header(line, line_start)
        if global_header is None:
            fields, header_problems = _header_fields(line, line_start)
        else:
            fields, header_problems = global_header
        problems.extend(header_problems)
        if not fields:
            continue
        kind = fields[0]
        if kind in (b"DATA", b"ECHO"):
            if len(fields) < 3:
                problems.append(
                    f"short {kind.decode(errors='replace')} header at "
                    f"byte {line_start}: {fields!r}")
                break
            try:
                size = _uint(fields[2],"payload size")
            except ValueError:
                problems.append(
                    f"invalid payload size at byte {line_start}: {fields!r}")
                break
            if size < 0:
                problems.append(
                    f"negative payload size at byte {line_start}: {fields!r}")
                break
            if size > n - i:
                problems.append(
                    f"truncated payload at byte {line_start}: need {size} bytes")
                break
            payload = blob[i:i + size]
            i += size
            if i < n and blob[i:i + 1] == b"\n":
                i += 1
            else:
                problems.append(
                    f"missing payload terminator at byte {line_start} "
                    f"after {fields!r}")
                break
            frames.append((fields, payload, line_start))
        else:
            frames.append((fields, None, line_start))
    return frames, problems


def capture_mode(frames):
    """Name the explicit transcript shape without grading its lifecycle."""
    if not frames:
        return "request-only"
    first = frames[0][0][0]
    if first == b"BANNER":
        return "full-process"
    if first == b"READY":
        return "ready-suffix"
    if any(fields[0] in _GLOBAL_KINDS for fields, _, _ in frames):
        return "invalid"
    return "request-only"


def _numeric_tail(fields, field_offset, expected_k, vocab, label, byte_offset):
    """Validate one DATA/ECHO numeric tail starting at fields[field_offset].

    Returns (problems, forms) where forms is the LIST of numeric-token
    "form" tags (see _numeric_value) observed in this tail, in order --
    a list, not a set, so the caller can tally counts. No rule requires a
    tail's forms to agree with one another: per-token classification is
    inherently ambiguous (a token can be an exact spelling under both
    finite grammars at once), so there is no reliable way to tell a
    genuinely mixed-format tail from an entirely single-format one that
    merely contains some ambiguous tokens -- see _numeric_value.
    """
    problems = []
    forms = []
    try:
        lp, lp_form = _numeric_value(fields[field_offset], f"{label} target logprob")
        k = _uint(fields[field_offset + 1], f"{label} top-k count", 32)
    except (ValueError, IndexError):
        return ([f"malformed {label} numeric fields at byte {byte_offset}: "
                f"{fields!r}"], forms)
    forms.append(lp_form)
    if not math.isfinite(lp) or lp > 0.0:
        problems.append(
            f"{label} target logprob is not finite/non-positive at byte "
            f"{byte_offset}: {fields!r}")
    if k != expected_k:
        problems.append(
            f"{label} top-k {k} != expected {expected_k} at byte "
            f"{byte_offset}: {fields!r}")
    want_fields = field_offset + 2 + 2 * k
    if len(fields) != want_fields:
        return (problems + [f"{label} field count {len(fields)} != "
                            f"{want_fields} at byte {byte_offset}: {fields!r}"],
                forms)
    ids = []
    for idx in range(k):
        try:
            token_id = _uint(fields[field_offset + 2 + 2 * idx],
                             f"{label} token id", _INT32_MAX)
            token_lp, token_form = _numeric_value(
                fields[field_offset + 3 + 2 * idx], f"{label} token logprob")
        except ValueError:
            problems.append(
                f"malformed {label} top-k pair {idx} at byte {byte_offset}: "
                f"{fields!r}")
            continue
        forms.append(token_form)
        if not 0 <= token_id < vocab:
            problems.append(
                f"{label} token id {token_id} outside [0,{vocab}) at byte "
                f"{byte_offset}: {fields!r}")
        if token_id in ids:
            problems.append(
                f"{label} duplicate token id {token_id} at byte "
                f"{byte_offset}: {fields!r}")
        ids.append(token_id)
        if not math.isfinite(token_lp) or token_lp > 0.0:
            problems.append(
                f"{label} token {token_id} logprob is not finite/non-positive "
                f"at byte {byte_offset}: {fields!r}")
    return problems, forms


def check(frames, framing_problems, request_id, topk, vocab):
    problems = list(framing_problems)
    mode = capture_mode(frames)
    if frames:
        first_kind = frames[0][0][0]
        if first_kind not in _GLOBAL_KINDS and first_kind not in _TARGETED_KINDS:
            problems.append(
                f"unrecognized frame opens the transcript at byte "
                f"{frames[0][2]} (not a global preamble or a request "
                f"frame): {frames[0][0]!r}")
    if mode == "invalid":
        problems.append(
            "invalid capture mode: global records present without a "
            "recognized BANNER/READY lead frame")
    try:
        rid = str(request_id).encode("ascii",errors="strict")
        rid_value=_uint(rid,"requested id",_UINT64_MAX)
        if rid_value < 1:
            raise ValueError("requested id must be positive")
    except (UnicodeEncodeError,ValueError) as exc:
        return 0, 0, problems+[str(exc)], mode, frozenset()
    data_frames = 0
    echo_positions = []
    accepts = []
    dones = []
    accept_prompts = []
    done_values = []
    done_seen = False
    data_seen = False
    global_indices = {kind: [] for kind in _GLOBAL_KINDS}
    expected_k = min(topk, vocab)
    forms_used = []
    for frame_index, (fields, payload, offset) in enumerate(frames):
        kind = fields[0]
        if kind in _GLOBAL_KINDS:
            global_indices[kind].append(frame_index)
            continue
        if len(fields)<2:
            continue
        try:
            frame_id=_uint(fields[1],"frame request id",_UINT64_MAX)
        except ValueError as exc:
            if kind in _TARGETED_KINDS:
                problems.append(f"{exc} at byte {offset}")
            continue
        if frame_id!=rid_value:
            continue
        if done_seen:
            problems.append(
                f"frame for id {request_id} after DONE at byte {offset}: "
                f"{fields!r}")
        if kind == b"ACCEPT":
            accepts.append(frame_index)
            if len(fields) != 3:
                problems.append(
                    f"malformed ACCEPT frame at byte {offset}: {fields!r}")
            else:
                try:
                    prompt=_uint(fields[2],"ACCEPT prompt length")
                    if prompt<1:
                        raise ValueError
                except ValueError:
                    problems.append(
                        f"invalid ACCEPT prompt length at byte {offset}: "
                        f"{fields!r}")
                else:
                    accept_prompts.append(prompt)
            continue
        if kind == b"ERROR":
            problems.append(
                f"target request returned ERROR at byte {offset}: {fields!r}")
            continue
        if kind == b"DONE":
            dones.append(frame_index)
            done_seen = True
            if len(fields) != 9 or fields[2] != b"STAT":
                problems.append(
                    f"malformed DONE frame at byte {offset}: {fields!r}")
                done_values.append(None)
                continue
            try:
                emitted=_uint(fields[3],"DONE emitted count")
                tps=_fixed_metric(fields[4],2,"DONE tokens/second")
                hit=_fixed_metric(fields[5],1,"DONE hit percentage",upper=100.0)
                rss=_fixed_metric(fields[6],2,"DONE RSS")
                prompt=_uint(fields[7],"DONE prompt-token count")
                flag=_uint(fields[8],"DONE length_limited",1)
                if prompt<1:
                    raise ValueError("DONE prompt-token count is not positive")
            except ValueError:
                problems.append(
                    f"malformed DONE stats at byte {offset}: {fields!r}")
                done_values.append(None)
                continue
            done_values.append((emitted,tps,hit,rss,prompt,flag))
            continue
        if kind == b"DATA":
            data_seen = True
            data_frames += 1
            if not accepts:
                problems.append(
                    f"DATA before ACCEPT at byte {offset}: {fields!r}")
            if payload is None:
                problems.append(
                    f"DATA missing validated payload at byte {offset}: "
                    f"{fields!r}")
            if len(fields) == 3:
                problems.append(
                    f"GAP: legacy 3-field DATA frame #{data_frames} at byte "
                    f"{offset} (payload {payload!r}) has no logprob")
                continue
            tail_problems, tail_forms = _numeric_tail(
                fields, 3, expected_k, vocab, "DATA", offset)
            problems.extend(tail_problems)
            forms_used.extend(tail_forms)
        elif kind == b"ECHO":
            if data_seen:
                problems.append(
                    f"ECHO after DATA at byte {offset}: {fields!r}")
            if not accepts:
                problems.append(
                    f"ECHO before ACCEPT at byte {offset}: {fields!r}")
            if payload is None:
                problems.append(
                    f"ECHO missing validated payload at byte {offset}: "
                    f"{fields!r}")
            try:
                pos = _uint(fields[3],"ECHO position")
            except (ValueError, IndexError):
                problems.append(
                    f"malformed ECHO frame at byte {offset}: {fields!r}")
                continue
            echo_positions.append(pos)
            if pos == 0:
                if len(fields) != 6 or fields[4] != b"nan" or fields[5] != b"0":
                    problems.append(
                        f"ECHO position 0 should carry exactly 'nan 0' at "
                        f"byte {offset}: {fields!r}")
            else:
                tail_problems, tail_forms = _numeric_tail(
                    fields, 4, expected_k, vocab, "ECHO", offset)
                problems.extend(tail_problems)
                forms_used.extend(tail_forms)
        else:
            problems.append(
                f"unknown targeted frame kind at byte {offset}: {fields!r}")
    any_globals = any(global_indices[kind] for kind in _GLOBAL_KINDS)
    if global_indices[b"BANNER"] or global_indices[b"LOADED"]:
        required = (
            (b"BANNER", 0), (b"LOADED", 1), (b"READY", 2), (b"STAT", 3),
            (b"HWINFO", 4), (b"TIERS", 5), (b"EMAP", 6),
        )
        for owned_kind, expected_index in required:
            found = global_indices[owned_kind]
            if (owned_kind in (b"BANNER", b"LOADED", b"READY", b"STAT") and
                    found != [expected_index]):
                problems.append(
                    f"expected one {owned_kind.decode()} as frame "
                    f"{expected_index}, found {found}")
            elif (owned_kind not in (b"BANNER", b"LOADED", b"READY", b"STAT") and
                  (not found or found[0] != expected_index)):
                problems.append(
                    f"expected startup {owned_kind.decode()} as frame "
                    f"{expected_index}, found {found}")
        if len(frames) > 7 and frames[7][0][0] not in (b"ACCEPT", b"ERROR"):
            problems.append(
                f"unexpected full-process pre-request frame 7: {frames[7][0]!r}")
    elif any_globals:
        ready = global_indices[b"READY"]
        stat = global_indices[b"STAT"]
        if ready != [0]:
            problems.append(f"expected one READY as frame 0, found {ready}")
        if stat != [1]:
            problems.append(f"expected one startup STAT as frame 1, found {stat}")
    if len(accepts) != 1:
        problems.append(f"expected exactly one ACCEPT, found {len(accepts)}")
    elif len(accept_prompts)==1:
        prompt=accept_prompts[0]
        if echo_positions!=list(range(prompt)):
            problems.append(f"ECHO positions {echo_positions} != required 0..{prompt-1}")
    if len(dones) != 1:
        problems.append(f"expected exactly one DONE, found {len(dones)}")
    elif len(done_values)==1 and done_values[0] is not None:
        emitted,_,_,_,done_prompt,_=done_values[0]
        if emitted!=data_frames:
            problems.append(f"DONE emitted {emitted} != observed DATA {data_frames}")
        if len(accept_prompts)==1 and done_prompt!=accept_prompts[0]:
            problems.append(f"DONE prompt count {done_prompt} != ACCEPT {accept_prompts[0]}")
    if data_frames<=0:
        problems.append("no DATA frames found for this id -- wrong id, or the run produced nothing")
    return data_frames, len(echo_positions), problems, mode, collections.Counter(forms_used)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("transcript", help="raw engine stdout capture")
    parser.add_argument("--id", required=True, help="request id to audit")
    parser.add_argument("--topk", type=int, required=True,
                        help="the SUBMIT logprobs=k value the request used")
    parser.add_argument("--vocab", type=int, required=True,
                        help="checkpoint vocabulary size used to bound token ids")
    args = parser.parse_args()
    try:
        if _uint(args.id.encode("ascii"), "requested id", _UINT64_MAX) < 1:
            raise ValueError("requested id must be positive")
    except (UnicodeEncodeError, ValueError) as exc:
        parser.error(str(exc))
    if not 1 <= args.topk <= 32:
        parser.error("--topk must be in 1..32 for an opted-in B1 evidence run")
    if args.vocab < 1:
        parser.error("--vocab must be positive")
    blob = open(args.transcript, "rb").read()
    frames, framing_problems = parse_frames(blob)
    data_frames, echo_frames, problems, mode, forms_used = check(
        frames, framing_problems, args.id, args.topk, args.vocab)
    # Informational only -- see _numeric_value: per-token form
    # classification is inherently ambiguous, so this tally never affects
    # the verdict, only what a reviewer sees.
    forms_summary = (", ".join(f"{form}={forms_used[form]}"
                               for form in sorted(forms_used))
                     if forms_used else "none")
    print(f"[gapcheck] id={args.id}: capture_mode={mode}, {data_frames} DATA "
          f"frames, {echo_frames} ECHO frames, numeric form tally: "
          f"{forms_summary}")
    for problem in problems:
        print(f"[gapcheck] {problem}")
    verdict = ("FAIL" if problems else
               "PASS: complete request; every generated token carries a valid logprob table")
    print(f"[gapcheck] {verdict}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
