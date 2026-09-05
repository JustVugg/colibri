import hashlib
import importlib.util
import json
import os
import tempfile
import unittest
import copy
from collections import Counter
from unittest.mock import patch
from pathlib import Path
import sys
import subprocess

ROOT = Path(__file__).resolve().parents[2]
C_ROOT = ROOT / "c"
TOOLS = C_ROOT / "tools"
sys.path.insert(0, str(TOOLS))
import mw_ur3_raw_adapter as adapter  # noqa: E402

INVALID_REQUEST_IDS = (
    "0", "+1", "01", "-1", str(adapter.REQUEST_ID_MAX + 1),
    " 1", "1 ", "\t1", "1\t", "one", "1x",
)

# The real frozen MW-UR3 authorities are opt-in only: set MW_UR3_CORPUS_DIR
# to a directory holding MW_UR3_CORPUS.jsonl, MW_UR3_EXPECTED_COMPARISONS.jsonl
# and MW_UR3_POLICY.json to exercise RealFrozenCorpusOptInTest below, the
# one test in this module allowed to depend on them. Every other test
# builds its own self-contained synthetic authority triple (below) so the
# substantive suite runs on any checkout with the variable unset.
_CORPUS_DIR = os.environ.get("MW_UR3_CORPUS_DIR")
AUTH = Path(_CORPUS_DIR) if _CORPUS_DIR else Path("/nonexistent-mw-ur3-corpus-dir")
REAL_CORPUS = AUTH / "MW_UR3_CORPUS.jsonl"
REAL_COMPARISONS = AUTH / "MW_UR3_EXPECTED_COMPARISONS.jsonl"
REAL_POLICY = AUTH / "MW_UR3_POLICY.json"
_REAL_AUTHORITY_SHA256 = dict(adapter.AUTHORITY_SHA256)


def _lcp(sequences):
    length = min(map(len, sequences))
    index = 0
    while index < length and len({sequence[index] for sequence in sequences}) == 1:
        index += 1
    return index


def _build_synthetic_authorities(directory):
    """Build a self-contained corpus/comparisons/policy triple that
    satisfies every structural constant :mod:`mw_ur3_raw_adapter` pins
    (EXPECTED's exact 20x4/240/56/24/80/20/420 census, BOUNDARY) without
    depending on the real frozen MW-UR3 objects.  20 items share a
    20-token prefix; items 0-13 are "normal" (context_length equals the
    shared prefix, contributing token_position + summed_score rows),
    items 14-19 are "semantically swallowed" (context_length short of the
    shared prefix by 2, contributing prefix_swallowed rows instead) --
    6*4=24 swallowed options, 14*4=56 normal options.  Item 0/option 0
    gets an exact 3-token continuation (positions 23/24/25) -- item 0's
    own shared prefix is 23 tokens (not the common 20), so position 25
    is what the token-position-boundary fixture mutates, and the
    order-sensitive-sum fixture indexes that continuation with a fixed
    3-entry tuple, so it must be exactly 3 long.  The remaining 55
    normal options are split 17-at-suffix-5 / 38-at-suffix-4 so the
    token_position census totals exactly 240 (3 + 17*5 + 38*4 = 240).
    """
    swallow_items = set(range(14, 20))
    normal_items = set(range(20)) - swallow_items
    non_zero_normal = [(item, option) for item in sorted(normal_items)
                       for option in range(4) if (item, option) != (0, 0)]
    assert len(non_zero_normal) == 55

    def prefix_length(item):
        # Item 0's shared prefix is 23 tokens (not the common-case 20) so
        # that item 0/option 0's exactly-3-token continuation (below) still
        # reaches absolute position 25 -- both the token-position-boundary
        # fixture (which mutates exactly item 0/option 0/position 25) and
        # the order-sensitive-sum fixture (which indexes item 0/option 0's
        # continuation with a fixed 3-entry tuple) target that one slot.
        return 23 if item == 0 else 20

    def suffix_length(item, option):
        if item in swallow_items:
            return 3
        if (item, option) == (0, 0):
            return 3
        return 5 if non_zero_normal.index((item, option)) < 17 else 4

    corpus_rows = []
    for item in range(20):
        prefix = [item * 1000 + i for i in range(prefix_length(item))]
        for option in range(4):
            continuation = [item * 1000 + 5000 + option * 100 + j
                            for j in range(suffix_length(item, option))]
            prompt_tokens = prefix + continuation
            context_length = (prefix_length(item) - 2 if item in swallow_items
                              else prefix_length(item))
            corpus_rows.append({
                "schema": "mw-ur3-corpus-v1", "item": item, "option": option,
                "context_length": context_length,
                "continuation_length": len(prompt_tokens) - context_length,
                "context_tokens": prompt_tokens[:context_length],
                "continuation_tokens": prompt_tokens[context_length:],
                "prompt_tokens": prompt_tokens,
            })
    corpus = {(row["item"], row["option"]): row for row in corpus_rows}
    lcps = {item: _lcp([corpus[item, option]["prompt_tokens"]
                        for option in range(4)]) for item in range(20)}
    assert all(lcps[item] == prefix_length(item) for item in range(20))

    expected_token = set(); expected_semantic = set(); expected_sum = set()
    for (item, option), row in corpus.items():
        ctx = row["context_length"]; detected = lcps[item]
        if detected > ctx:
            expected_semantic.add((item, option, ctx, detected, detected - ctx))
        else:
            expected_token.update(
                (item, option, position, row["prompt_tokens"][position])
                for position in range(detected, len(row["prompt_tokens"])))
            expected_sum.add(
                (item, option, tuple(range(ctx, len(row["prompt_tokens"])))))
    assert len(expected_token) == 240
    assert len(expected_semantic) == 24
    assert len(expected_sum) == 56

    comparison_rows = []
    for item, option, position, token in sorted(expected_token):
        comparison_rows.append({
            "schema": "mw-ur3-comparison-v1", "kind": "token_position",
            "item": item, "option": option, "absolute_position": position,
            "token_id": token,
        })
    for item, option, ctx, detected, not_returned in sorted(expected_semantic):
        comparison_rows.append({
            "schema": "mw-ur3-comparison-v1", "kind": "prefix_swallowed",
            "item": item, "option": option, "context_length": ctx,
            "detected_prefix_length": detected,
            "continuation_positions_not_returned": not_returned,
        })
    for item, option, positions in sorted(expected_sum):
        comparison_rows.append({
            "schema": "mw-ur3-comparison-v1", "kind": "summed_score",
            "item": item, "option": option, "positions": list(positions),
        })
    for item in range(20):
        for option in range(4):
            row = corpus[item, option]
            prompt = row["prompt_tokens"]; context = row["context_length"]
            comparison_rows.append({
                "schema": "mw-ur3-comparison-v1", "kind": "option_argmax_input",
                "item": item, "option": option,
                "group_absolute_positions": list(range(lcps[item], len(prompt))),
                "sequential_absolute_positions": list(range(context, len(prompt))),
                "prefix_swallowed_positions": max(0, lcps[item] - context),
            })
    for item in range(20):
        comparison_rows.append({
            "schema": "mw-ur3-comparison-v1", "kind": "item_argmax",
            "item": item, "options": [0, 1, 2, 3],
            "option_input_identities": [[item, option] for option in range(4)],
        })
    assert len(comparison_rows) == 420

    policy = {
        "schema": "mw-ur3-policy-v1",
        "required_comparisons": {
            "token_positions": 240, "summed_scores": 56,
            "prefix_swallowed_semantics": 24, "option_argmax_inputs": 80,
            "item_argmax": 20,
        },
        "historical_delta_boundary": adapter.BOUNDARY,
    }

    corpus_path = directory / "synthetic_corpus.jsonl"
    comparisons_path = directory / "synthetic_comparisons.jsonl"
    policy_path = directory / "synthetic_policy.json"
    corpus_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in corpus_rows) + "\n",
        encoding="utf-8")
    comparisons_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in comparison_rows) + "\n",
        encoding="utf-8")
    policy_path.write_text(json.dumps(policy, sort_keys=True), encoding="utf-8")
    return corpus_path, comparisons_path, policy_path


_SYNTH_DIR = tempfile.TemporaryDirectory(prefix="mw_ur3_synthetic_authorities_")
CORPUS, COMPARISONS, POLICY = _build_synthetic_authorities(Path(_SYNTH_DIR.name))
_SYNTH_AUTHORITY_SHA256 = {
    "corpus": adapter.sha256(CORPUS), "comparisons": adapter.sha256(COMPARISONS),
    "policy": adapter.sha256(POLICY),
}
_authority_patch = patch.object(adapter, "AUTHORITY_SHA256", _SYNTH_AUTHORITY_SHA256)


def setUpModule():
    _authority_patch.start()


def tearDownModule():
    _authority_patch.stop()
    _SYNTH_DIR.cleanup()


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def lcp(sequences):
    length = min(map(len, sequences))
    index = 0
    while index < length and len({sequence[index] for sequence in sequences}) == 1:
        index += 1
    return index


def lp(item, option, position, group=False, boundary=False, flip=False):
    value = -(0.01 + option * 0.001 + (position % 7) * 0.00001)
    if group and boundary and item == 0 and option == 0 and position == 25:
        value -= 0.001
    if group and flip and item == 0 and option == 3:
        value = -0.000001
    return value


def numeric(value):
    return format(value, ".17g")


def token_payload(token):
    return f"<token:{token}>".encode()


def request_record(tokens):
    return " ".join(str(token) for token in tokens).encode("ascii")


def payload_frame(header, payload):
    fields = header.split(" ")
    fields[2] = str(len(payload))
    return " ".join(fields).encode() + b"\n" + payload + b"\n"


def tail(value, token, topk):
    pairs = [(token + offset, value - float(offset))
             for offset in range(topk - 1, 0, -1)]
    pairs.append((token, value))
    return (f"{numeric(value)} {topk} " +
            " ".join(f"{tid} {numeric(score)}" for tid, score in pairs))


def make_bundle(directory, *, boundary=False, flip=False,
                tie=False, sum_boundary=False, drop_group_echo=False,
                order_sensitive=False,
                duplicate_group_echo=False, reorder_group_echo=False,
                misattribute_grpg=False, bad_grpe_greedy=False,
                drop_score=False, missing_sequential_topk=False,
                bad_prof=False, bad_done=False,
                bad_hash=False, bad_evidence_hash=False,
                bad_request_hash=False, bad_token_binding=False,
                bad_capture_hash=False, request_payload_mismatch=False,
                payload_mismatch=False, final_payload_mismatch=False,
                bad_grpg_target=False,
                drop_sequential_data=False, extra_sequential_data=False,
                bad_data_shape=False, bad_data_payload=False,
                bad_data_target=False, bad_sequential_prof=False,
                bad_sequential_done=False, request_max_tokens=1,
                first_group_id=None, topk=1, forge_unsorted_greedy=False,
                forged_token_payload_authority=False,
                corpus_path=None, comparisons_path=None, policy_path=None):
    corpus_path = corpus_path or CORPUS
    comparisons_path = comparisons_path or COMPARISONS
    policy_path = policy_path or POLICY
    corpus_rows = records(corpus_path)
    corpus = {(row["item"], row["option"]): row for row in corpus_rows}
    group_ids = {str(item): str(1000 + item) for item in range(20)}
    if first_group_id is not None:
        group_ids["0"] = first_group_id
    sequential_ids = {f"{item}:{option}": str(2000 + item * 4 + option)
                      for item in range(20) for option in range(4)}
    group_request_sha256 = {}
    sequential_request_sha256 = {}
    group_request_frames = []
    sequential_request_frames = []
    for item in range(20):
        prompts = [corpus[item, option]["prompt_tokens"] for option in range(4)]
        detected = lcp(prompts)
        wire = b"\n".join(
            [request_record(prompts[0][:detected])] +
            [request_record(prompt[detected:]) for prompt in prompts])
        captured = wire
        if request_payload_mismatch and item == 0:
            captured = b"4" + wire[1:]
        group_request_sha256[str(item)] = hashlib.sha256(captured).hexdigest()
        group_request_frames.append(
            f"SUBMIT {group_ids[str(item)]} 0 {len(captured)} "
            f"{request_max_tokens} 0 1 0 "
            f"ids=1 logprobs={topk} group=4\n".encode() + captured + b"\n")
        for option, prompt in enumerate(prompts):
            sequential_wire = request_record(prompt)
            sequential_request_sha256[f"{item}:{option}"] = \
                hashlib.sha256(sequential_wire).hexdigest()
            sequential_request_frames.append(
                f"SUBMIT {sequential_ids[f'{item}:{option}']} 0 "
                f"{len(sequential_wire)} {request_max_tokens} 0 1 0 "
                f"ids=1 logprobs={topk}\n".encode() +
                sequential_wire + b"\n")
    observed_tokens = {token for row in corpus_rows
                       for token in row["prompt_tokens"]}
    observed_tokens.add(0)  # the synthetic final-distribution argmax
    manifest = {
        "schema": adapter.RUN_SCHEMA,
        "candidate_head": "a" * 40,
        "candidate_tree": "b" * 40,
        "build_id": "synthetic-build",
        "container_id": "synthetic-container",
        "model_id": "synthetic-model",
        "route_id": "explicit-group-and-ordinary-sequential",
        "run_id": "synthetic-run",
        "authority_sha256": {
            "corpus": adapter.sha256(corpus_path),
            "comparisons": adapter.sha256(comparisons_path),
            "policy": adapter.sha256(policy_path),
        },
        "group_request_ids": group_ids,
        "sequential_request_ids": sequential_ids,
        "logprobs": {"group": topk, "sequential": topk},
        "request_sha256": {
            "group": group_request_sha256,
            "sequential": sequential_request_sha256,
        },
        "token_payload_sha256": {
            str(token): hashlib.sha256(token_payload(token)).hexdigest()
            for token in sorted(observed_tokens)
        },
    }
    if bad_hash:
        manifest["authority_sha256"]["corpus"] = "0" * 64
    if bad_request_hash:
        manifest["request_sha256"]["group"]["0"] = "0" * 64
    if bad_token_binding:
        manifest["token_payload_sha256"][str(min(observed_tokens))] = "0" * 64

    group_frames = []
    dropped = False; payload_changed = False; final_payload_changed = False
    grpg_target_changed = False
    for item in range(20):
        rid = group_ids[str(item)]
        prompts = [corpus[item, option]["prompt_tokens"] for option in range(4)]
        detected = lcp(prompts)
        suffix_lengths = [len(prompt) - detected for prompt in prompts]
        group_frames.append(f"ACCEPT {rid} {detected + sum(suffix_lengths)}\n".encode())
        for position in range(detected):
            token = prompts[0][position]
            if position == 0:
                group_frames.append(payload_frame(
                    f"GRPP {rid} 0 0 nan 0", token_payload(token)))
            else:
                value = lp(item, 0, position)
                group_frames.append(payload_frame(
                    f"GRPP {rid} 0 {position} {tail(value, token, topk)}",
                    token_payload(token)))
        for option, prompt in enumerate(prompts):
            length = len(prompt) - detected
            group_frames.append(f"GRPS {rid} {option} {detected} {length}\n".encode())
            values = []
            for position in range(detected, len(prompt)):
                if tie:
                    value = -0.5 if position == detected else 0.0
                elif order_sensitive and item == 0 and option == 0:
                    value = (-1.0e16, -1.0, -1.0)[position - detected]
                else:
                    value = lp(item, option, position, True, boundary, flip)
                    if sum_boundary:
                        value -= 1.0e-5
                values.append(value); token = prompt[position]
                payload = token_payload(token)
                if payload_mismatch and not payload_changed:
                    payload = b"X" * len(payload); payload_changed = True
                frame = payload_frame(
                    f"ECHO {rid} 0 {position} {tail(value, token, topk)}",
                    payload)
                if drop_group_echo and not dropped:
                    dropped = True
                else:
                    group_frames.append(frame)
                    if duplicate_group_echo and not dropped:
                        group_frames.append(frame); dropped = True
            final = -0.02
            final_payload = token_payload(0)
            if final_payload_mismatch and not final_payload_changed:
                final_payload = b"Y" * len(final_payload)
                final_payload_changed = True
            final_tail = tail(final, 0, topk)
            if bad_grpg_target and not grpg_target_changed:
                final_tail = numeric(final - 1.0) + " " + final_tail.split(" ", 1)[1]
                grpg_target_changed = True
            group_frames.append(payload_frame(
                f"GRPG {rid} 0 {option} {len(prompt)} {final_tail}",
                final_payload))
            wrong_greedy = ((bad_grpe_greedy or forge_unsorted_greedy) and
                            item == 0 and option == 0)
            group_frames.append(
                f"GRPE {rid} {option} {numeric(adapter.ltr(values))} {length} "
                f"{0 if wrong_greedy else 1}\n".encode())
        group_frames.append(
            f"PROF {'nan' if bad_prof and item == 0 else '1.000'} "
            f"{detected + sum(suffix_lengths)} 0 0.000 0.000 0.000 0.000 0.000 0\n".encode())
        group_frames.append(
            f"DONE {rid} STAT 0 0.00 0.0 1.00 {detected + sum(suffix_lengths)} "
            f"{2 if bad_done and item == 0 else 0}\n".encode())

    sequential_frames = []
    score_lines = []
    for ordinal, row in enumerate(corpus_rows):
        item, option = row["item"], row["option"]
        rid = sequential_ids[f"{item}:{option}"]
        prompt = row["prompt_tokens"]
        sequential_frames.append(f"ACCEPT {rid} {len(prompt)}\n".encode())
        for position, token in enumerate(prompt):
            if position == 0:
                sequential_frames.append(payload_frame(
                    f"ECHO {rid} 0 0 nan 0", token_payload(token)))
            else:
                if tie and position >= row["context_length"]:
                    value = -0.5 if position == row["context_length"] else 0.0
                elif order_sensitive and item == 0 and option == 0 and \
                        position >= row["context_length"]:
                    value = (-1.0e16, -1.0, -1.0)[position - row["context_length"]]
                else:
                    value = lp(item, option, position)
                if missing_sequential_topk and item == 0 and option == 0 and \
                        position == row["context_length"]:
                    sequential_frames.append(payload_frame(
                        f"ECHO {rid} 0 {position} {numeric(value)} 0",
                        token_payload(token)))
                else:
                    sequential_frames.append(payload_frame(
                        f"ECHO {rid} 0 {position} {tail(value, token, topk)}",
                        token_payload(token)))
        data_value = -1.02 if bad_data_target and item == 0 and option == 0 else -0.02
        data_payload = (b"Z" * len(token_payload(0))
                        if bad_data_payload and item == 0 and option == 0
                        else token_payload(0))
        data_tail = tail(-0.02, 0, topk)
        if bad_data_target and item == 0 and option == 0:
            data_tail = data_tail.replace(numeric(-0.02), numeric(data_value), 1)
        data_shape_prefix = "0 " if bad_data_shape and item == 0 and option == 0 else ""
        data_frame = payload_frame(
            f"DATA {rid} 0 {data_shape_prefix}{data_tail}", data_payload)
        if not (drop_sequential_data and item == 0 and option == 0):
            sequential_frames.append(data_frame)
        if extra_sequential_data and item == 0 and option == 0:
            sequential_frames.append(payload_frame(
                f"DATA {rid} 0 {tail(-0.02, 0, topk)}", token_payload(0)))
        sequential_frames.append(
            f"PROF 1.000 {len(prompt)} "
            f"{0 if bad_sequential_prof and item == 0 and option == 0 else 1} "
            f"0.000 0.000 0.000 0.000 0.000 0\n".encode())
        sequential_frames.append(
            f"DONE {rid} STAT "
            f"{0 if bad_sequential_done and item == 0 and option == 0 else 1} "
            f"0.00 0.0 1.00 {len(prompt)} "
            f"{0 if bad_sequential_done and item == 0 and option == 0 else 1}\n".encode())
        request_wire = (f"{row['context_length']} {row['continuation_length']} " +
                        " ".join(map(str, prompt)) + "\n").encode()
        digest = hashlib.sha256(request_wire).hexdigest()
        values = [((-0.5 if position == row["context_length"] else 0.0) if tie else
                   ((-1.0e16, -1.0, -1.0)[position - row["context_length"]]
                    if order_sensitive and item == 0 and option == 0 else
                    lp(item, option, position)))
                  for position in range(row["context_length"], len(prompt))]
        score_lines.append(
            f"SCORE {ordinal} {digest} {numeric(adapter.ltr(values))} "
            f"{row['continuation_length']} "
            f"{0 if forge_unsorted_greedy and item == 0 and option == 0 else 1}\n")
    if drop_score:
        score_lines.pop()
    if reorder_group_echo:
        indices = [index for index, frame in enumerate(group_frames)
                   if frame.startswith(b"ECHO 1000 ")]
        group_frames[indices[0]], group_frames[indices[1]] = (
            group_frames[indices[1]], group_frames[indices[0]])
    if misattribute_grpg:
        prefix = f"GRPG 1000 {len(token_payload(0))} 0 ".encode()
        index = next(index for index, frame in enumerate(group_frames)
                     if frame.startswith(prefix))
        group_frames[index] = group_frames[index].replace(
            prefix, f"GRPG 1000 {len(token_payload(0))} 1 ".encode(), 1)

    manifest_path = directory / "run.json"
    tokenizer_path = directory / "tokenizer.json"
    container_path = directory / "model.safetensors.index.json"
    group_requests_path = directory / "group_requests.raw"
    sequential_requests_path = directory / "sequential_requests.raw"
    group_path = directory / "group.raw"
    sequential_path = directory / "sequential.raw"
    score_path = directory / "score.txt"
    output_path = directory / "result.json"
    tokenizer_path.write_text(json.dumps({
        "model": {"type": "BPE", "vocab": {}, "merges": []},
        "added_tokens": [
            {"id": token, "content": token_payload(token).decode("ascii"),
             "special": False}
            for token in sorted(observed_tokens)
        ],
    }, sort_keys=True), encoding="utf-8", newline="")
    container_path.write_bytes(b'{"synthetic":"container-manifest"}\n')
    group_requests_path.write_bytes(b"".join(group_request_frames))
    sequential_requests_path.write_bytes(b"".join(sequential_request_frames))
    group_blob = b"".join(group_frames)
    sequential_blob = b"".join(sequential_frames)
    if forged_token_payload_authority:
        canonical = token_payload(0)
        forged = b"X" * len(canonical)
        group_blob = group_blob.replace(canonical, forged)
        sequential_blob = sequential_blob.replace(canonical, forged)
        manifest["token_payload_sha256"]["0"] = hashlib.sha256(forged).hexdigest()
    group_path.write_bytes(group_blob)
    sequential_path.write_bytes(sequential_blob)
    score_path.write_text("".join(score_lines), encoding="ascii", newline="")
    manifest["capture_sha256"] = {
        "tokenizer": adapter.sha256(tokenizer_path),
        "container_manifest": adapter.sha256(container_path),
    }
    manifest["evidence_sha256"] = {
        "group_frames": adapter.sha256(group_path),
        "sequential_frames": adapter.sha256(sequential_path),
        "score_evidence": adapter.sha256(score_path),
        "group_requests": adapter.sha256(group_requests_path),
        "sequential_requests": adapter.sha256(sequential_requests_path),
    }
    if bad_capture_hash:
        manifest["capture_sha256"]["tokenizer"] = "0" * 64
    if bad_evidence_hash:
        manifest["evidence_sha256"]["group_frames"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return [
        "--corpus", str(corpus_path), "--comparisons", str(comparisons_path),
        "--policy", str(policy_path), "--run-manifest", str(manifest_path),
        "--tokenizer", str(tokenizer_path),
        "--container-manifest", str(container_path),
        "--group-requests", str(group_requests_path),
        "--sequential-requests", str(sequential_requests_path),
        "--group-frames", str(group_path),
        "--sequential-frames", str(sequential_path),
        "--score-evidence", str(score_path), "--output", str(output_path),
    ], output_path


class TokenizerPayloadTest(unittest.TestCase):
    def test_tokenizer_decoder_is_independent_bytelevel_authority(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "tokenizer.json"
            path.write_text(json.dumps({
                "model": {"type": "BPE", "vocab": {
                    "A": 1,
                    chr(256): 2,
                    "B" * 300: 3,
                }, "merges": []},
                "added_tokens": [{"id": 4, "content": "é", "special": False}],
            }), encoding="utf-8", newline="")
            self.assertEqual(adapter.tokenizer_payloads(
                    path.read_bytes(), {1, 2, 3, 4}, str(path)), {
                1: b"A",
                2: b"\x00",
                3: b"B" * 256,
                4: "é".encode("utf-8"),
            })


class AuthorityBindingTest(unittest.TestCase):
    """Pin the raw-file authority check independent of manifest tampering:
    a corpus/comparisons/policy file that is not byte-identical to the
    frozen MW-UR3 objects must be refused even when nothing in the run
    manifest claims otherwise (require_bindings checks the files
    themselves, not just the manifest's declared digest of them)."""

    def test_non_frozen_authority_file_is_refused(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        directory = Path(temporary.name)
        argv, _ = make_bundle(directory)
        manifest = json.loads(
            Path(argv[argv.index("--run-manifest") + 1]).read_text(
                encoding="utf-8"))
        corrupt_corpus = directory / "corpus.jsonl"
        corrupt_corpus.write_bytes(CORPUS.read_bytes() + b"\n")
        with self.assertRaisesRegex(
                adapter.EvidenceError, "frozen MW-UR3 objects"):
            adapter.require_bindings(
                manifest,
                {"corpus": corrupt_corpus, "comparisons": COMPARISONS,
                 "policy": POLICY},
                {"tokenizer": Path(argv[argv.index("--tokenizer") + 1]),
                 "container_manifest":
                     Path(argv[argv.index("--container-manifest") + 1])},
                {"group_frames": Path(argv[argv.index("--group-frames") + 1]),
                 "sequential_frames":
                     Path(argv[argv.index("--sequential-frames") + 1]),
                 "score_evidence":
                     Path(argv[argv.index("--score-evidence") + 1]),
                 "group_requests":
                     Path(argv[argv.index("--group-requests") + 1]),
                 "sequential_requests":
                     Path(argv[argv.index("--sequential-requests") + 1])})


@unittest.skipUnless(
    REAL_CORPUS.exists(),
    "real frozen MW-UR3 corpus not available: set MW_UR3_CORPUS_DIR to a "
    "directory holding MW_UR3_CORPUS.jsonl, MW_UR3_EXPECTED_COMPARISONS.jsonl "
    "and MW_UR3_POLICY.json to run this test; every other test in this "
    "module is self-contained and does not need it")
class RealFrozenCorpusOptInTest(unittest.TestCase):
    """The ONE test in this module allowed to depend on the real frozen
    MW-UR3 authorities.  Everything else builds its own synthetic
    corpus/comparisons/policy triple (see _build_synthetic_authorities
    above) so the substantive suite runs without MW_UR3_CORPUS_DIR set."""

    def test_real_frozen_corpus_binds_and_passes(self):
        with patch.object(adapter, "AUTHORITY_SHA256", _REAL_AUTHORITY_SHA256):
            temporary = tempfile.TemporaryDirectory()
            self.addCleanup(temporary.cleanup)
            argv, output = make_bundle(
                Path(temporary.name), corpus_path=REAL_CORPUS,
                comparisons_path=REAL_COMPARISONS, policy_path=REAL_POLICY)
            status = adapter.main(argv)
        result = json.loads(output.read_text())
        self.assertEqual(status, 0)
        self.assertEqual(result["verdict"], "PASS")


class ReadOnceHashingTest(unittest.TestCase):
    """TOCTOU: every bound file must be hashed and parsed from the same
    single read, never a hash-then-reopen-and-parse pair -- otherwise a
    file swapped between the two reads lets content that never matched
    its own declared digest through undetected."""

    TRACKED_FLAGS = (
        "run-manifest", "corpus", "comparisons", "policy",
        "group-frames", "sequential-frames", "score-evidence",
        "group-requests", "sequential-requests",
    )

    def test_every_bound_file_is_read_exactly_once(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        argv, output = make_bundle(Path(temporary.name))
        tracked = {Path(argv[argv.index(f"--{flag}") + 1])
                  for flag in self.TRACKED_FLAGS}
        counts = Counter()
        real_read_bytes = Path.read_bytes

        def spy(self):
            if self in tracked:
                counts[self] += 1
            return real_read_bytes(self)

        with patch.object(Path, "read_bytes", spy):
            status = adapter.main(argv)
        result = json.loads(output.read_text())
        self.assertEqual(status, 0)
        self.assertEqual(result["verdict"], "PASS")
        for path in tracked:
            with self.subTest(path=path.name):
                self.assertEqual(counts[path], 1)

    def test_swapping_the_run_manifest_after_its_hash_cannot_change_output(self):
        """Direct demonstration of the exploit the fix closes: a fake
        opener that would hand back a tampered manifest on any read past
        the first.  Old (double-read) code hashes the real bytes but
        parses the swapped ones; the fix must never issue that second
        read, so the result reflects only the real, hashed content."""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        argv, output = make_bundle(Path(temporary.name))
        manifest_path = Path(argv[argv.index("--run-manifest") + 1])
        original_bytes = manifest_path.read_bytes()
        original_manifest = json.loads(original_bytes)
        swapped_manifest = dict(original_manifest)
        swapped_manifest["run_id"] = "swapped-by-attacker"
        swapped_bytes = json.dumps(swapped_manifest).encode("utf-8")
        self.assertNotEqual(hashlib.sha256(swapped_bytes).hexdigest(),
                            hashlib.sha256(original_bytes).hexdigest())
        real_read_bytes = Path.read_bytes
        calls = []

        def fake_opener(self):
            if self == manifest_path:
                calls.append(1)
                return original_bytes if len(calls) == 1 else swapped_bytes
            return real_read_bytes(self)

        with patch.object(Path, "read_bytes", fake_opener):
            status = adapter.main(argv)
        result = json.loads(output.read_text())
        self.assertEqual(len(calls), 1,
                         "a second read would have returned the swapped "
                         "manifest; the fix must never issue it")
        self.assertEqual(status, 0)
        self.assertEqual(result["verdict"], "PASS")
        self.assertEqual(result["bindings"]["run_id"], original_manifest["run_id"])


class MissingAuthorityTest(unittest.TestCase):
    """A missing bound file must surface as a named failure identifying
    the role and a relative path -- never a raw FileNotFoundError with an
    absolute host path, and never the generic internal-failure BLOCK path
    that previously swallowed it."""

    ROLE_FLAGS = (
        ("run manifest", "run-manifest"),
        ("corpus", "corpus"),
        ("comparisons", "comparisons"),
        ("policy", "policy"),
        ("tokenizer", "tokenizer"),
        ("container_manifest", "container-manifest"),
    )

    def test_missing_file_is_a_named_failure_per_role(self):
        for role, flag in self.ROLE_FLAGS:
            with self.subTest(role=role):
                temporary = tempfile.TemporaryDirectory()
                self.addCleanup(temporary.cleanup)
                argv, output = make_bundle(Path(temporary.name))
                index = argv.index(f"--{flag}") + 1
                missing = Path(temporary.name) / "does-not-exist.missing"
                argv[index] = str(missing)

                status = adapter.main(argv)
                result = json.loads(output.read_text())
                self.assertEqual(status, 1)
                self.assertEqual(result["verdict"], "FAIL")
                self.assertIn(f"missing authority: {role} (", result["disposition"])
                self.assertNotIn(str(temporary.name), result["disposition"])
                self.assertNotIn(temporary.name, result["disposition"])


class FieldOffsetTest(unittest.TestCase):
    """A field-level error inside a frame (not just a truncated/malformed
    header) must carry that frame's byte offset, same as header errors
    already do -- lets an operator find the exact byte in a multi-megabyte
    capture without a separate line-count pass."""

    def test_malformed_field_carries_frame_byte_offset(self):
        frame = {"kind": "ACCEPT", "fields": ["ACCEPT", "1", "bogus"],
                 "payload": None, "offset": 123}
        with self.assertRaisesRegex(adapter.EvidenceError, r"^byte 123: "):
            adapter.correlate_group([frame], {0: "1"}, {}, {0: 0}, 1)

    def test_sequential_field_error_also_carries_offset(self):
        frame = {"kind": "ACCEPT", "fields": ["ACCEPT", "1", "bogus"],
                 "payload": None, "offset": 456}
        with self.assertRaisesRegex(adapter.EvidenceError, r"^byte 456: "):
            adapter.correlate_sequential([frame], {(0, 0): "1"}, {}, 1)


class LcpLtrLiteralTest(unittest.TestCase):
    """Pin ``lcp``/``ltr`` against hand-computed literal expectations,
    not values the fixture builder derives by calling the same functions
    (that would be tautological -- a broken lcp/ltr would still "agree"
    with a fixture built from its own output). Includes empty and
    single-element inputs."""

    def test_lcp_literal_cases(self):
        self.assertEqual(adapter.lcp([[1, 2, 3], [1, 2, 4]]), 2)
        self.assertEqual(adapter.lcp([[1, 2, 3], [1, 2, 3]]), 3)
        self.assertEqual(adapter.lcp([[1], [2]]), 0)
        self.assertEqual(adapter.lcp([[], []]), 0)
        self.assertEqual(adapter.lcp([[1, 2], [1, 2, 3]]), 2)
        self.assertEqual(adapter.lcp([[7]]), 1)
        self.assertEqual(adapter.lcp([[1, 2, 3], [1, 2, 3], [1, 9, 3]]), 1)

    def test_ltr_literal_cases(self):
        self.assertEqual(adapter.ltr([]), 0.0)
        self.assertEqual(adapter.ltr([1.0]), 1.0)
        self.assertEqual(adapter.ltr([1.0, 2.0, 3.0]), 6.0)
        self.assertEqual(adapter.ltr([-1.0, -2.0]), -3.0)
        self.assertEqual(adapter.ltr([0.1, 0.2]), 0.1 + 0.2)
        # Order-sensitive by construction, hand-computed: left-to-right,
        # 1.0 + 1e16 rounds to 1e16 (1.0 is below the ulp=2 at that
        # magnitude), then + -1e16 cancels exactly to 0.0. Summed in the
        # opposite direction (-1e16 + 1e16 == 0.0, then + 1.0 == 1.0) the
        # result is 1.0 instead -- this case is not reversal-invariant, so
        # it fails at the unit level under a right-to-left mutation.
        self.assertEqual(adapter.ltr([1.0, 1e16, -1e16]), 0.0)
        with self.assertRaises(adapter.EvidenceError):
            adapter.ltr([float("nan")])
        with self.assertRaises(adapter.EvidenceError):
            adapter.ltr([float("inf")])
        with self.assertRaises(adapter.EvidenceError):
            adapter.ltr([None])


class NumericGrammarTest(unittest.TestCase):
    """Every numeric field accepts both the %.6f and %.17g forms, pinned
    directly against a non-dyadic value (0.1 has no exact binary64
    %.17g short form, unlike e.g. 0.5)."""

    def test_logprob_accepts_both_canonical_forms(self):
        value = -0.1
        six = format(value, ".6f")
        seventeen = format(value, ".17g")
        self.assertNotEqual(six, seventeen)
        self.assertEqual(adapter.logprob(six, "test"), float(six))
        self.assertEqual(adapter.logprob(seventeen, "test"), float(seventeen))

    def test_tail_accepts_both_canonical_forms(self):
        value = -0.1
        for text in (format(value, ".6f"), format(value, ".17g")):
            with self.subTest(text=text):
                target, token, best, k = adapter.validate_tail(
                    [text, "1", "7", text], 0, "test")
                self.assertEqual(target, float(text))
                self.assertEqual(token, 7)
                self.assertEqual(best, float(text))
                self.assertEqual(k, 1)


class RawAdapterTest(unittest.TestCase):
    def run_bundle(self, **kwargs):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        argv, output = make_bundle(Path(temporary.name), **kwargs)
        status = adapter.main(argv)
        return status, json.loads(output.read_text())

    def test_exact_denominators_and_raw_pass(self):
        status, result = self.run_bundle()
        self.assertEqual(status, 0)
        self.assertEqual(result["verdict"], "PASS")
        self.assertEqual(result["denominators"], adapter.EXPECTED)
        self.assertEqual(len(result["token_positions"]), 240)
        self.assertEqual(len(result["summed_scores"]), 56)
        self.assertEqual(len(result["prefix_swallowed_semantics"]), 24)
        self.assertEqual(len(result["option_argmax_inputs"]), 80)
        self.assertEqual(len(result["items"]), 20)
        self.assertEqual(len(result["final_distributions"]), 80)
        self.assertEqual(len(result["sequential_generations"]), 80)
        self.assertEqual(result["request_profile"]["max_tokens"], 1)
        self.assertEqual(result["argmax_flips"], 0)
        self.assertRegex(result["run_manifest_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(set(result["capture_sha256"]),
                         {"tokenizer", "container_manifest"})

    def test_request_ids_are_exact_positive_uint64(self):
        for value in ("1", str(adapter.REQUEST_ID_MAX)):
            with self.subTest(valid=value):
                self.assertEqual(adapter.request_id(value, "test request id"), value)
                status, result = self.run_bundle(first_group_id=value)
                self.assertEqual(status, 0)
                self.assertEqual(result["verdict"], "PASS")
        for value in INVALID_REQUEST_IDS:
            with self.subTest(invalid=value):
                with self.assertRaises(adapter.EvidenceError):
                    adapter.request_id(value, "test request id")
                status, result = self.run_bundle(first_group_id=value)
                self.assertEqual(status, 1)
                self.assertEqual(result["verdict"], "FAIL")
                self.assertIn("request id", result["disposition"])

        group = {str(item): str(1000 + item) for item in range(20)}
        sequential = {f"{item}:{option}": str(2000 + item * 4 + option)
                      for item in range(20) for option in range(4)}
        for value in INVALID_REQUEST_IDS:
            with self.subTest(invalid_group_manifest=value):
                bad_group = dict(group); bad_group["0"] = value
                with self.assertRaisesRegex(adapter.EvidenceError, "request id"):
                    adapter.request_maps({
                        "group_request_ids": bad_group,
                        "sequential_request_ids": sequential,
                    })
            with self.subTest(invalid_sequential_manifest=value):
                bad_sequential = dict(sequential); bad_sequential["0:0"] = value
                with self.assertRaisesRegex(adapter.EvidenceError, "request id"):
                    adapter.request_maps({
                        "group_request_ids": group,
                        "sequential_request_ids": bad_sequential,
                    })

    @unittest.skip(
        "needs tests/test_group_score, which this build does not produce"
    )
    def test_actual_production_parser_dispatcher_frames_are_accepted(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        directory = Path(temporary.name)
        paths = [directory / name for name in (
            "group.requests", "group.frames",
            "sequential.requests", "sequential.frames")]
        subprocess.run(["make", "tests/test_group_score"], cwd=C_ROOT,
                       check=True, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, text=True)
        subprocess.run([
            str(C_ROOT / "tests" / "test_group_score"),
            "--emit-adapter-lifecycle", *(str(path) for path in paths),
        ], cwd=C_ROOT, check=True, stdout=subprocess.PIPE,
           stderr=subprocess.PIPE)

        group_requests = adapter.parse_request_capture(
            paths[0], group=True, expected_topk=2)
        sequential_requests = adapter.parse_request_capture(
            paths[2], group=False, expected_topk=2)
        self.assertEqual(group_requests,
                         [("1", b"3 1 4\n5\n6\n7\n8")])
        self.assertEqual(sequential_requests,
                         [(str(adapter.REQUEST_ID_MAX), b"3 1 4 5")])

        group_corpus = {
            (0, option): {"prompt_tokens": [3, 1, 4, 5 + option]}
            for option in range(4)
        }
        group_frames = adapter.parse_frames(paths[1])
        sequential_frames = adapter.parse_frames(paths[3])
        group_counts = Counter(frame["kind"] for frame in group_frames)
        sequential_counts = Counter(frame["kind"] for frame in sequential_frames)
        self.assertEqual({kind: group_counts[kind] for kind in (
            "ACCEPT", "GRPP", "GRPS", "ECHO", "GRPG", "GRPE", "PROF", "DONE")},
            {"ACCEPT": 1, "GRPP": 3, "GRPS": 4, "ECHO": 4,
             "GRPG": 4, "GRPE": 4, "PROF": 1, "DONE": 1})
        self.assertEqual(group_counts["DATA"], 0)
        self.assertEqual({kind: sequential_counts[kind] for kind in (
            "ACCEPT", "ECHO", "DATA", "PROF", "DONE")},
            {"ACCEPT": 1, "ECHO": 4, "DATA": 1, "PROF": 1, "DONE": 1})

        group = adapter.correlate_group(
            group_frames, {0: "1"}, group_corpus,
            {0: 3}, 2)
        sequential = adapter.correlate_sequential(
            sequential_frames, {(0, 0): str(adapter.REQUEST_ID_MAX)},
            {(0, 0): group_corpus[0, 0]}, 2)
        self.assertEqual(len(group[0]), 4)
        self.assertEqual(len(group[4]), 4)
        self.assertEqual(len(sequential[0]), 4)
        self.assertEqual(set(sequential[3]), {(0, 0)})
        self.assertEqual(sequential[3][0, 0]["ordinal"], 0)

        request_kinds = {
            "ACCEPT", "DONE", "GRPP", "GRPS", "ECHO", "GRPG", "GRPE",
            "DATA", "ERROR",
        }
        for index, value in enumerate(INVALID_REQUEST_IDS):
            with self.subTest(invalid_capture=value):
                bad_capture = directory / f"bad-{index}.requests"
                bad_capture.write_bytes(paths[0].read_bytes().replace(
                    b"SUBMIT 1 ", f"SUBMIT {value} ".encode(), 1))
                with self.assertRaisesRegex(
                        adapter.EvidenceError, r"request (?:header|id)"):
                    adapter.parse_request_capture(
                        bad_capture, group=True, expected_topk=2)

            with self.subTest(invalid_group_frame=value):
                bad_group = copy.deepcopy(group_frames)
                for frame in bad_group:
                    if frame["kind"] in request_kinds:
                        frame["fields"][1] = value
                with self.assertRaisesRegex(adapter.EvidenceError, "request id"):
                    adapter.correlate_group(
                        bad_group, {0: value}, group_corpus, {0: 3}, 2)

            with self.subTest(invalid_sequential_frame=value):
                bad_sequential = copy.deepcopy(sequential_frames)
                for frame in bad_sequential:
                    if frame["kind"] in request_kinds:
                        frame["fields"][1] = value
                with self.assertRaisesRegex(adapter.EvidenceError, "request id"):
                    adapter.correlate_sequential(
                        bad_sequential, {(0, 0): value},
                        {(0, 0): group_corpus[0, 0]}, 2)

    def test_missing_group_identity_fails_not_partial_pass(self):
        status, result = self.run_bundle(drop_group_echo=True)
        self.assertEqual(status, 1)
        self.assertEqual(result["verdict"], "FAIL")
        self.assertRegex(result["disposition"], r"out-of-order|shape")

    def test_unknown_frame_kind_is_refused_by_name(self):
        """An unrecognized wire kind is a named refusal, never
        silently skipped. Pinned independently for both frame streams."""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        argv, output = make_bundle(Path(temporary.name))
        group_path = Path(argv[argv.index("--group-frames") + 1])
        manifest_path = Path(argv[argv.index("--run-manifest") + 1])
        group_path.write_bytes(b"BOGUS 1\n" + group_path.read_bytes())
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["evidence_sha256"]["group_frames"] = adapter.sha256(group_path)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        status = adapter.main(argv)
        result = json.loads(output.read_text())
        self.assertEqual(status, 1)
        self.assertEqual(result["verdict"], "FAIL")
        self.assertIn("unexpected raw group frame: BOGUS", result["disposition"])

        temporary2 = tempfile.TemporaryDirectory()
        self.addCleanup(temporary2.cleanup)
        argv2, output2 = make_bundle(Path(temporary2.name))
        sequential_path = Path(argv2[argv2.index("--sequential-frames") + 1])
        manifest_path2 = Path(argv2[argv2.index("--run-manifest") + 1])
        sequential_path.write_bytes(
            b"BOGUS 1\n" + sequential_path.read_bytes())
        manifest2 = json.loads(manifest_path2.read_text(encoding="utf-8"))
        manifest2["evidence_sha256"]["sequential_frames"] = adapter.sha256(
            sequential_path)
        manifest_path2.write_text(json.dumps(manifest2), encoding="utf-8")

        status2 = adapter.main(argv2)
        result2 = json.loads(output2.read_text())
        self.assertEqual(status2, 1)
        self.assertEqual(result2["verdict"], "FAIL")
        self.assertIn("unexpected raw sequential frame: BOGUS",
                      result2["disposition"])

    def test_missing_score_identity_fails(self):
        status, result = self.run_bundle(drop_score=True)
        self.assertEqual(status, 1)
        self.assertEqual(result["verdict"], "FAIL")
        self.assertIn("80-row", result["disposition"])

    def test_scored_row_without_required_topk_fails(self):
        status, result = self.run_bundle(missing_sequential_topk=True)
        self.assertEqual(status, 1)
        self.assertEqual(result["verdict"], "FAIL")
        self.assertRegex(result["disposition"], r"scored row has no top-k|wrong top-k count")

    def test_malformed_prof_and_done_cannot_pass(self):
        for mutation in ({"bad_prof": True}, {"bad_done": True}):
            with self.subTest(mutation=mutation):
                status, result = self.run_bundle(**mutation)
                self.assertEqual(status, 1)
                self.assertEqual(result["verdict"], "FAIL")

    def test_capture_profile_rejects_engine_invalid_zero_max_tokens(self):
        status, result = self.run_bundle(request_max_tokens=0)
        self.assertEqual(status, 1)
        self.assertEqual(result["verdict"], "FAIL")
        self.assertIn("wrong MW-UR3 request profile", result["disposition"])

    def test_one_generation_lifecycle_is_exact(self):
        for mutation in ({"drop_sequential_data": True},
                         {"extra_sequential_data": True},
                         {"bad_data_shape": True},
                         {"bad_data_payload": True},
                         {"bad_data_target": True},
                         {"bad_sequential_prof": True},
                         {"bad_sequential_done": True}):
            with self.subTest(mutation=mutation):
                status, result = self.run_bundle(**mutation)
                self.assertEqual(status, 1)
                self.assertEqual(result["verdict"], "FAIL")

    def test_authority_hash_drift_fails(self):
        status, result = self.run_bundle(bad_hash=True)
        self.assertEqual(status, 1)
        self.assertEqual(result["verdict"], "FAIL")
        self.assertIn("authority_sha256", result["disposition"])

    def test_raw_evidence_hash_drift_fails(self):
        status, result = self.run_bundle(bad_evidence_hash=True)
        self.assertEqual(status, 1)
        self.assertIn("evidence_sha256", result["disposition"])

    def test_exact_request_byte_binding_is_required(self):
        status, result = self.run_bundle(bad_request_hash=True)
        self.assertEqual(status, 1)
        self.assertEqual(result["verdict"], "FAIL")
        self.assertIn("request-byte binding", result["disposition"])

    def test_honestly_rehashed_wrong_request_capture_cannot_pass(self):
        status, result = self.run_bundle(request_payload_mismatch=True)
        self.assertEqual(status, 1)
        self.assertEqual(result["verdict"], "FAIL")
        self.assertIn("request-byte binding", result["disposition"])

    def test_tokenizer_and_container_capture_hashes_are_required(self):
        status, result = self.run_bundle(bad_capture_hash=True)
        self.assertEqual(status, 1)
        self.assertEqual(result["verdict"], "FAIL")
        self.assertIn("capture_sha256", result["disposition"])

    def test_token_payload_authority_drift_fails(self):
        status, result = self.run_bundle(bad_token_binding=True)
        self.assertEqual(status, 1)
        self.assertEqual(result["verdict"], "FAIL")
        self.assertIn("token payload identity", result["disposition"])

    def test_internally_consistent_wrong_payload_is_not_tokenizer_authority(self):
        status, result = self.run_bundle(forged_token_payload_authority=True)
        self.assertEqual(status, 1)
        self.assertEqual(result["verdict"], "FAIL")
        self.assertIn("tokenizer-derived", result["disposition"])

    def test_tokenizer_path_replacement_after_hash_cannot_change_authority(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        args, output_path = make_bundle(
            Path(temporary.name), forged_token_payload_authority=True)
        tokenizer_index = args.index("--tokenizer") + 1
        tokenizer_path = Path(args[tokenizer_index])
        original = adapter.require_bindings

        def replace_after_binding(*positional, **keywords):
            result = original(*positional, **keywords)
            tokenizer = json.loads(tokenizer_path.read_text(encoding="utf-8"))
            record = next(record for record in tokenizer["added_tokens"]
                          if record["id"] == 0)
            record["content"] = "X" * len(token_payload(0))
            tokenizer_path.write_text(json.dumps(tokenizer, sort_keys=True),
                                      encoding="utf-8", newline="")
            return result

        with patch.object(adapter, "require_bindings",
                          side_effect=replace_after_binding):
            status = adapter.main(args)
        result = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(status, 1)
        self.assertEqual(result["verdict"], "FAIL")
        self.assertIn("tokenizer-derived", result["disposition"])

    def test_group_sequential_payload_mismatch_cannot_pass(self):
        status, result = self.run_bundle(payload_mismatch=True)
        self.assertEqual(status, 1)
        self.assertEqual(result["verdict"], "FAIL")
        self.assertRegex(result["disposition"], r"payload identity|payload mismatch")

    def test_final_distribution_payload_mismatch_cannot_pass(self):
        status, result = self.run_bundle(final_payload_mismatch=True)
        self.assertEqual(status, 1)
        self.assertEqual(result["verdict"], "FAIL")
        self.assertIn("group final", result["disposition"])

    def test_final_distribution_target_must_be_argmax(self):
        status, result = self.run_bundle(bad_grpg_target=True)
        self.assertEqual(status, 1)
        self.assertEqual(result["verdict"], "FAIL")
        self.assertIn("GRPG target is not its argmax", result["disposition"])

    def test_unsorted_topk_recovers_argmax_by_value(self):
        status, result = self.run_bundle(topk=2)
        self.assertEqual(status, 0)
        self.assertEqual(result["verdict"], "PASS")
        self.assertEqual(result["logprobs"], {"group": 2, "sequential": 2})

    def test_unsorted_first_entry_cannot_forge_greedy(self):
        status, result = self.run_bundle(topk=2, forge_unsorted_greedy=True)
        self.assertEqual(status, 1)
        self.assertEqual(result["verdict"], "FAIL")
        self.assertRegex(result["disposition"], r"GRPE sum/shape|SCORE/ECHO")

    def test_unexpected_input_failure_overwrites_stale_pass_fail_closed(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        argv, output = make_bundle(Path(temporary.name))
        output.write_text('{"verdict":"PASS","stale":true}\n')
        manifest = Path(argv[argv.index("--run-manifest") + 1])
        manifest.write_text("{", encoding="utf-8")
        status = adapter.main(argv)
        result = json.loads(output.read_text())
        self.assertEqual(status, 2)
        self.assertEqual(result["verdict"], "BLOCK")
        self.assertNotIn("stale", result)

    def test_duplicate_run_manifest_key_fails_closed(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        argv, output = make_bundle(Path(temporary.name))
        manifest = Path(argv[argv.index("--run-manifest") + 1])
        wire = manifest.read_text(encoding="utf-8")
        manifest.write_text('{"run_id":"shadow",' + wire[1:], encoding="utf-8")
        status = adapter.main(argv)
        result = json.loads(output.read_text())
        self.assertEqual(status, 1)
        self.assertEqual(result["verdict"], "FAIL")
        self.assertIn("duplicate JSON key", result["disposition"])

    def test_group_duplicate_reorder_misattribution_and_greedy_bite(self):
        for mutation in ({"duplicate_group_echo": True},
                         {"reorder_group_echo": True},
                         {"misattribute_grpg": True},
                         {"bad_grpe_greedy": True}):
            with self.subTest(mutation=mutation):
                status, result = self.run_bundle(**mutation)
                self.assertEqual(status, 1)
                self.assertEqual(result["verdict"], "FAIL")

    def test_exact_ties_choose_lowest_option_index(self):
        status, result = self.run_bundle(tie=True)
        self.assertEqual(status, 0)
        self.assertEqual(result["verdict"], "PASS")
        self.assertTrue(all(row["group_argmax"] == 0 and
                            row["sequential_argmax"] == 0 and
                            row["group_margin"] == 0.0 and
                            row["sequential_margin"] == 0.0
                            for row in result["items"]))

    def test_sum_deltas_do_not_use_token_position_boundary(self):
        status, result = self.run_bundle(sum_boundary=True)
        self.assertEqual(status, 0)
        self.assertEqual(result["verdict"], "PASS")
        self.assertLessEqual(result["maximum_absolute_token_position_delta"],
                             adapter.BOUNDARY)
        self.assertGreater(max(abs(row["delta"]) for row in result["summed_scores"]),
                           adapter.BOUNDARY)

    def test_left_to_right_binary64_accumulation_is_load_bearing(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        argv, output = make_bundle(Path(temporary.name), order_sensitive=True)

        def right_to_left(values):
            total = 0.0
            for value in reversed(list(values)):
                total += value
            return total

        with patch.object(adapter, "ltr", right_to_left):
            status = adapter.main(argv)
        result = json.loads(output.read_text())
        self.assertEqual(status, 1)
        self.assertEqual(result["verdict"], "FAIL")
        self.assertRegex(result["disposition"], r"GRPE sum|SCORE/ECHO")

    def test_prefix_swallowed_rows_remain_raw_semantics(self):
        status, result = self.run_bundle()
        self.assertEqual(status, 0)
        rows = result["prefix_swallowed_semantics"]
        self.assertEqual(len(rows), 24)
        self.assertTrue(any(row["shared_prefix_logprobs"] !=
                            row["sequential_logprobs"] for row in rows))

    def test_token_position_boundary_is_stop_not_tolerance(self):
        status, result = self.run_bundle(boundary=True)
        self.assertEqual(status, 3)
        self.assertEqual(result["verdict"], "STOP")
        self.assertGreater(result["maximum_absolute_token_position_delta"],
                           adapter.BOUNDARY)

    def test_argmax_flip_is_fail(self):
        status, result = self.run_bundle(flip=True)
        self.assertEqual(status, 1)
        self.assertEqual(result["verdict"], "FAIL")
        self.assertGreater(result["argmax_flips"], 0)

    def test_internal_authority_denominator_and_identity_mutations_fail(self):
        corpus_rows = records(CORPUS)
        comparison_rows = records(COMPARISONS)
        policy = json.loads(POLICY.read_text())
        for kind in ("token_position", "summed_score", "prefix_swallowed",
                     "option_argmax_input", "item_argmax"):
            mutated = copy.deepcopy(comparison_rows)
            mutated.pop(next(index for index, row in enumerate(mutated)
                             if row["kind"] == kind))
            with self.subTest(missing_kind=kind), self.assertRaises(adapter.EvidenceError):
                adapter.validate_authorities(corpus_rows, mutated, policy)

        duplicated = copy.deepcopy(comparison_rows)
        first = next(row for row in duplicated if row["kind"] == "token_position")
        victim = next(index for index, row in enumerate(duplicated)
                      if row["kind"] == "token_position" and row is not first)
        duplicated[victim] = copy.deepcopy(first)
        with self.assertRaisesRegex(adapter.EvidenceError, "duplicate"):
            adapter.validate_authorities(corpus_rows, duplicated, policy)

        swapped = copy.deepcopy(corpus_rows)
        for left, right in ((swapped[0], swapped[1]),):
            li, ri = left["context_length"], right["context_length"]
            left["continuation_tokens"][0], right["continuation_tokens"][0] = (
                right["continuation_tokens"][0], left["continuation_tokens"][0])
            left["prompt_tokens"][li] = left["continuation_tokens"][0]
            right["prompt_tokens"][ri] = right["continuation_tokens"][0]
        with self.assertRaises(adapter.EvidenceError):
            adapter.validate_authorities(swapped, comparison_rows, policy)

        shrunken_policy = copy.deepcopy(policy)
        shrunken_policy["required_comparisons"]["token_positions"] -= 1
        shrunken = copy.deepcopy(comparison_rows)
        shrunken.pop(next(index for index, row in enumerate(shrunken)
                          if row["kind"] == "token_position"))
        with self.assertRaisesRegex(adapter.EvidenceError, "denominator"):
            adapter.validate_authorities(corpus_rows, shrunken, shrunken_policy)

    def _blank_evidence_file(self, argv, flag):
        """Empty one evidence file in place and re-bind its manifest digest
        so the bundle stays otherwise internally consistent -- used to
        synthesize the "this engine build never emits this wire kind"
        captures the per-family tests below need."""
        path = Path(argv[argv.index(f"--{flag}") + 1])
        path.write_bytes(b"")
        manifest_path = Path(argv[argv.index("--run-manifest") + 1])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        key = flag.replace("-", "_")
        manifest["evidence_sha256"][key] = adapter.sha256(path)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    def test_family_support_direct(self):
        """Unit-level pin on the helper itself, independent of the CLI
        path: it reports presence per family and never raises."""
        empty_score = b""
        one_score = b"SCORE 0 " + b"a" * 64 + b" -0.1 1 1\n"
        self.assertEqual(adapter.family_support([], empty_score), (False, False))
        self.assertEqual(
            adapter.family_support(
                [{"kind": "GRPP", "fields": [], "payload": None, "offset": 0}],
                empty_score),
            (True, False))
        self.assertEqual(adapter.family_support([], one_score), (False, True))
        self.assertEqual(
            adapter.family_support(
                [{"kind": "GRPG", "fields": [], "payload": None, "offset": 0}],
                one_score),
            (True, True))

    def test_neither_family_supported_is_honestly_unsupported(self):
        """Combination 1/3 (none): a capture from an engine that
        never emits group frames or score-evidence lines must not produce
        a silent pass, nor a plain FAIL indistinguishable from a malformed
        capture -- it gets its own named verdict, and neither family ran."""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        argv, output = make_bundle(Path(temporary.name))
        self._blank_evidence_file(argv, "group-frames")
        self._blank_evidence_file(argv, "score-evidence")

        status = adapter.main(argv)
        result = json.loads(output.read_text())
        self.assertEqual(status, 4)
        self.assertEqual(result["verdict"], "UNSUPPORTED")
        self.assertEqual(result["group_checks"], "UNSUPPORTED")
        self.assertEqual(result["score_checks"], "UNSUPPORTED")
        self.assertEqual(result["checks_ran"], [])
        self.assertIn("group frame kinds", result["disposition"])

    def test_score_evidence_only_reports_group_unsupported(self):
        """Combination 2/3 (evidence-only): score-evidence lines
        present, group frames absent -- a capture that emits
        score-evidence lines but no group frames. Must report
        UNSUPPORTED for the group family (not FAIL,
        and not an overall PASS), while the score family still runs."""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        argv, output = make_bundle(Path(temporary.name))
        self._blank_evidence_file(argv, "group-frames")

        status = adapter.main(argv)
        result = json.loads(output.read_text())
        self.assertEqual(status, 4)
        self.assertEqual(result["verdict"], "UNSUPPORTED")
        self.assertEqual(result["group_checks"], "UNSUPPORTED")
        self.assertEqual(result["score_checks"], "PASS")
        self.assertEqual(result["checks_ran"], ["score"])

    def test_group_frames_only_reports_score_unsupported(self):
        """Symmetric variant (group present, score-evidence absent):
        the group family runs and passes on its own denominators, but the
        overall verdict still cannot be a clean PASS since the score
        family never ran."""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        argv, output = make_bundle(Path(temporary.name))
        self._blank_evidence_file(argv, "score-evidence")

        status = adapter.main(argv)
        result = json.loads(output.read_text())
        self.assertEqual(status, 4)
        self.assertEqual(result["verdict"], "UNSUPPORTED")
        self.assertEqual(result["group_checks"], "PASS")
        self.assertEqual(result["score_checks"], "UNSUPPORTED")
        self.assertEqual(result["checks_ran"], ["group"])
        self.assertEqual(len(result["token_positions"]), 240)

    def test_both_families_supported_is_the_only_path_to_pass(self):
        """Combination 3/3 (both): the ordinary case every other test in
        this module already exercises through run_bundle() -- restated
        here as the explicit per-family contract: only two families that
        both ran and passed produce group_checks == score_checks == PASS
        and the overall PASS."""
        status, result = self.run_bundle()
        self.assertEqual(status, 0)
        self.assertEqual(result["verdict"], "PASS")
        self.assertEqual(result["group_checks"], "PASS")
        self.assertEqual(result["score_checks"], "PASS")
        self.assertEqual(result["checks_ran"], ["group", "score"])


class AuthoritySha256LiteralPinTest(unittest.TestCase):
    """The module's real-authority binding (AUTHORITY_SHA256) is exercised
    end to end only by RealFrozenCorpusOptInTest above, which skips
    whenever MW_UR3_CORPUS_DIR is not set -- true on every clean
    checkout. This test needs no corpus: it pins the three digests the
    module ships against literals stored here, independently of the
    module's own source, so a silent edit to either side is caught even
    with the real objects unavailable.

    The `adapter` name imported at the top of this file has its
    AUTHORITY_SHA256 replaced with a synthetic triple for the whole
    module's test run (setUpModule/tearDownModule, above), so this test
    loads a second, never-patched copy of the same file directly rather
    than reading through that name."""

    EXPECTED = {
        "corpus": "d750ab07d611a140836f70ccba31fe28e586630d365131c4ce52e63120a07b6e",
        "comparisons": "b4c73d22a0d5f7e595f7833f8b31c4669bafbc928fa21192b585df74455f3476",
        "policy": "308734a31af638e912ccec790df07704d4d0f1d82b253353bddaa39aa261c56f",
    }

    @staticmethod
    def _load_unpatched_adapter():
        spec = importlib.util.spec_from_file_location(
            "mw_ur3_raw_adapter_unpatched", TOOLS / "mw_ur3_raw_adapter.py")
        fresh = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fresh)
        return fresh

    def test_authority_sha256_matches_pinned_literals(self):
        fresh = self._load_unpatched_adapter()
        self.assertEqual(fresh.AUTHORITY_SHA256, self.EXPECTED)

    def test_every_pinned_literal_is_64_lowercase_hex_characters(self):
        for role, digest in self.EXPECTED.items():
            with self.subTest(role=role):
                self.assertEqual(len(digest), 64)
                self.assertRegex(digest, r"^[0-9a-f]{64}$")


if __name__ == "__main__":
    unittest.main()
