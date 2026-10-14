#!/usr/bin/env python3
"""Write laya_tiny/ref.json: what the `laya` package answers on the tiny fixture.

Maintainer-run, offline, once per change of the fixture: it needs torch,
transformers and the pinned `laya` package (pip install laya==0.3.24), which CI
does not. The answers come from Convai Innovations' own runtime, laya.Agent, on
the checkpoint tools/make_laya_tiny.py writes; the SHA-256 of that checkpoint is
recorded so tests/test_laya_tiny.py refuses a regenerated fixture that differs.

For every case it records, per question, what the engine must reproduce:
  ids, markers        Agent._encode_state (common.build_sequence)
  logits              DecisionModel.forward, the raw scorer output
  probs               Agent._decode_answers' calibrated softmax, unrounded
  act_probability     the act head's softmax, entry 0
and the package's own rounded answer (Agent.predict), against which the
unrounded numbers are checked here before anything is written. A case the
package refuses records the refusal instead. Every decision is checked for a
near-tie, so a float-level difference can never flip an argmax in the test.

    python tools/make_laya_tiny.py --output ./laya_tiny
    python tools/make_laya_ref.py --fixture ./laya_tiny
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

PINNED_LAYA = "0.3.24"
MIN_GAP = 1e-3                  # smallest top-1 / top-2 probability gap accepted

LONG_BODY = (
    "Hello team, I am writing again about the invoice we received on the first of the month. "
    "The amount is twice what we agreed in the contract and the payment was already taken from "
    "our card. We asked for a refund last week and nobody answered. Our finance department needs "
    "the corrected invoice before Friday or we will have to escalate this to our lawyers and stop "
    "using the service. Please also check the other invoices of this year, because we think the "
    "same mistake happened in March and in April. Thank you, the accounts team.")

CONVERSATION = [
    {"role": "user", "content": "Hi, I cannot log in to my account since yesterday."},
    {"role": "assistant", "content": "Sorry about that. Did you try to reset the password?"},
    {"role": "user", "content": "Yes, twice, and the reset email never arrives."},
    {"role": "assistant", "content": "Let me check the email address on the account."},
    {"role": "user", "content": "It is the same as always. This is really urgent, I have a "
                                "payment due today and I need the invoices from the portal."},
    {"role": "assistant", "content": "I understand. I am escalating it now."},
    {"role": "user", "content": "If it is not fixed today I will cancel the subscription."},
]

DEPARTMENT = {"type": "choice", "instructions": "Which department should handle this request?",
              "criteria": {"billing": "invoices, payments, refunds",
                           "technical": "bugs, outages, system errors",
                           "sales": "pricing, new contracts", "other": "everything else"}}

CASES = [
    ("readme_ticket",
     {"from": "user@acme.com", "subject": "Duplicate charge on invoice #4411",
      "body": "Hi, we were billed twice for March. Please refund the duplicate today or we will "
              "cancel our plan."},
     {"department": DEPARTMENT,
      "urgency": {"type": "score", "instructions": "How urgent is this request?",
                  "criteria": ["not urgent", "soon", "critical deadline or blocking issue"]},
      "churn_risk": {"type": "noul", "instructions": "Does the user threaten to cancel or leave?"},
      "refund_requested": {"type": "noul",
                           "instructions": "Does the user explicitly request a refund?"}}),
    ("plain_two_options",
     "The application crashes every time I open the settings page.",
     {"is_bug": {"type": "choice", "instructions": "Is this a bug report or a feature request?",
                 "criteria": {"bug": "something is broken", "feature": "something new is wanted"}},
      "blocking": {"type": "noul", "instructions": "Is the user blocked?",
                   "criteria": {"true": "the user cannot work at all",
                                "false": "the user can still work"}},
      "only_true": {"type": "noul", "instructions": "Does it mention settings?",
                    "criteria": {"true": "the settings page is named"}}}),
    ("choice_seven_structured",
     "We would like to schedule a demo of the product for our team next week.",
     {"intent": {"type": "choice", "instructions": "What does the sender want?",
                 "criteria": {"demo": "a product demonstration", "refund": None, "cancel": "",
                              "pricing": {"desc": "price questions", "examples": ["cost", "plan"]},
                              "support": 3, "partnership": ["co-marketing", "reselling"],
                              "other": "anything else"}}}),
    ("choice_twelve_labels",
     "I love this product, it works great and the support team was very helpful.",
     {"emotion": {"type": "choice", "instructions": "Which emotion does the text express?",
                  "criteria": {name: None for name in (
                      "joy", "anger", "sadness", "fear", "surprise", "disgust", "trust",
                      "anticipation", "neutral", "confusion", "gratitude", "pride")}}}),
    ("scores",
     "This is the worst service I have ever used, nothing works and nobody answers.",
     {"sentiment": {"type": "score", "instructions": "What is the sender's tone?",
                    "criteria": ["angry", "negative", "neutral", "positive", "delighted"]},
      "six_levels": {"type": "score", "instructions": "How severe is the impact?",
                     "criteria": ["none", "minor", "moderate", "major", "critical", "catastrophic"]},
      "two_levels": {"type": "score", "instructions": "Is any action needed?",
                     "criteria": ["no action", "action needed"]},
      "numeric_levels": {"type": "score", "instructions": "Rate the urgency.",
                         "criteria": [1, 2, 3]}}),
    ("long_state_truncated",
     {"subject": "Invoice problem again", "body": LONG_BODY},
     {"department": DEPARTMENT,
      "legal_threat": {"type": "noul", "instructions": "Does the sender threaten legal action?"}}),
    ("conversation_truncated_left", CONVERSATION,
     {"churn": {"type": "noul", "instructions": "Does the user threaten to cancel?"},
      "topic": {"type": "choice", "instructions": "What is the conversation about?",
                "criteria": {"login": "access to the account", "billing": "payments",
                             "other": "anything else"}}}),
    ("json_instructions_and_state",
     {"ticket": {"id": 4411, "amount": 129.5, "currency": "EUR", "paid": True,
                 "refund": None, "ratio": 1e-05, "big": 1e21, "tags": ["billing", "urgent"],
                 "nested": {"a": [1, 2.0, {"b": "c"}]}}},
     {"action": {"type": "choice",
                 "instructions": {"task": "pick the next action", "context": "support desk",
                                  "steps": ["read", "decide"]},
                 "criteria": {"refund": "give the money back", "escalate": "a human decides",
                              "close": "nothing to do"}},
      "paid": {"type": "noul", "instructions": ["Was", "the", "ticket", "paid?"]}}),
    ("special_text",
     "Order [MASK] arrived  broken,   see\tphoto.\r\n\r\nThe café [SEP] said |||IP_ADDRESS||| "
     "is blocked <|endoftext|> and [unused0] support said 'we're sorry'.",
     {"damaged": {"type": "noul", "instructions": "Was the item [MASK] damaged?"},
      "channel": {"type": "choice", "instructions": "Which channel  should   answer?",
                  "criteria": {"email": "write an email [MASK] back",
                               "phone": "call\tthe customer", "none": None}}}),
    ("long_option",
     "The meeting notes say that the contract will be renewed if the price stays the same.",
     {"renewal": {"type": "choice", "instructions": "Will the contract be renewed?",
                  "criteria": {"yes": "the contract will be renewed " + "under the same terms " * 12,
                               "no": "the contract ends", "unclear": "the notes do not say"}}}),
    ("many_options_shrunk",
     "Security incident: a phishing email asked employees to reset their passwords.",
     {"category": {"type": "choice",
                   "instructions": "Classify the incident into exactly one of the categories "
                                   "below, considering the attack vector, the target and the "
                                   "likely impact on the organisation and its customers.",
                   "criteria": {f"cat{i:02d}": f"category number {i} of the incident taxonomy"
                                for i in range(20)}}}),
    ("too_many_options",
     "Too many options for the head budget.",
     {"bucket": {"type": "choice", "instructions": "Pick one.",
                 "criteria": {f"option{i}": f"the description of option {i}" for i in range(40)}}}),
    ("unicode",
     "Buongiorno, la fattura è sbagliata. Grüße aus München, ευχαριστώ. "
     "我们的账户被锁了 \U0001F642\U0001F44D\U0001F3FD Å Ω",
     {"language": {"type": "choice", "instructions": "Which language comes first?",
                   "criteria": {"italian": "italiano", "german": "deutsch", "chinese": "中文"}},
      "polite": {"type": "noul", "instructions": "Is the message polite?"}}),
]

TOKENIZER_STRINGS = [
    "", " ", "  ", "     ", "\n\n", "\t", "a  b   c    d         e", "Hello world",
    "don't I'll we've they're she'd 'S 'T", "2026-10-02 3.14 1000000 12345678901",
    "café café Å Ω Å", "日本語のテキスト",
    "\U0001F642\U0001F44D\U0001F3FD", "[SEP] in [CLS] text [PAD][UNK]",
    "|||IP_ADDRESS||| and |||EMAIL_ADDRESS|||, |||PHONE_NUMBER|||", "<|endoftext|><|padding|>",
    "[unused0]x[unused1]", "tab\tand\r\nnewline\n", "   leading spaces", "trailing spaces   ",
    "mixed ,.;:!? punctuation!!! ... ???", "über straße Ελληνικά "
    "русский عربي", "x" * 300,
    "{\"subject\": \"Duplicate charge\", \"body\": \"Please fix this\"}",
    "level 0: not urgent", " choice question: Which department?",
]


def compact_json(ref):
    """One key per line at the top, one case or tokenizer string per line below
    it: diffable, without a line per token id."""
    def dump(value):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    lines = ["{"]
    keys = list(ref)
    for i, key in enumerate(keys):
        tail = "," if i + 1 < len(keys) else ""
        value = ref[key]
        if isinstance(value, list):
            lines.append(f" {dump(key)}:[")
            lines.extend(f"  {dump(item)}{',' if j + 1 < len(value) else ''}"
                         for j, item in enumerate(value))
            lines.append(f" ]{tail}")
        else:
            lines.append(f" {dump(key)}:{dump(value)}{tail}")
    lines.append("}")
    return "\n".join(lines) + "\n"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fixture", default="./laya_tiny")
    parser.add_argument("--allow-version", action="store_true",
                        help=f"run with an installed laya other than {PINNED_LAYA}")
    args = parser.parse_args()

    import torch
    import transformers
    import laya
    from laya.common import QTYPES, collate_items, temp_bucket

    if laya.__version__ != PINNED_LAYA and not args.allow_version:
        sys.exit(f"laya {laya.__version__} is installed; the reference is pinned to {PINNED_LAYA}")
    torch.manual_seed(0)
    fixture = Path(args.fixture)
    agent = laya.load(str(fixture), device="cpu")
    pad = agent.tok.pad_token_id

    cases = []
    for name, state, questions in CASES:
        case = {"name": name, "state": state, "questions": questions}
        ids = list(questions)
        try:
            for qid in ids:
                agent._check_question(qid, questions[qid])
            internal = {qid: agent._to_internal(questions[qid]) for qid in ids}
            items = agent._encode_state(state, ids, internal)
        except ValueError as error:
            case["error"] = str(error)
            cases.append(case)
            print(f"{name}: refused ({error})", file=sys.stderr)
            continue
        batch = collate_items([items], pad)
        with torch.no_grad():
            logits, act = agent._forward(batch)
        predicted = agent.predict(state, questions)
        answers = []
        for j, qid in enumerate(ids):
            q = internal[qid]
            k = len(items[j]["markers"])
            qt = QTYPES[q["t"]]
            # Agent._decode_answers, unrounded
            t_scale = agent.temperature_by_options.get(temp_bucket(qt, k), agent.temperature[qt])
            z = logits[j, :k] / t_scale
            p = np.exp(z - z.max())
            p = p / p.sum()
            order = np.sort(p)[::-1]
            gap = float(order[0] - order[1]) if k > 1 else 1.0
            if gap < MIN_GAP:
                sys.exit(f"{name}.{qid}: top-2 gap {gap:.2e} is a near-tie; change the case or SEED")
            got = predicted["answers"][qid]
            reference = (got["probabilities"].values() if q["t"] != "noul" else [1 - got["noul"], got["noul"]])
            if q["t"] != "noul" and any(abs(round(float(a), 4) - b) > 1e-12 for a, b in zip(p, reference)):
                sys.exit(f"{name}.{qid}: unrounded probabilities disagree with Agent.predict")
            if q["t"] == "noul" and abs(round(float(p[1]), 4) - got["noul"]) > 1e-12:
                sys.exit(f"{name}.{qid}: unrounded noul disagrees with Agent.predict")
            answers.append({
                "id": qid, "ids": [int(x) for x in items[j]["ids"]],
                "markers": [int(x) for x in items[j]["markers"]],
                "logits": [float(x) for x in logits[j, :k]],
                "probs": [float(x) for x in p], "temperature": float(t_scale),
                "act_probability": float(act[j, 0]),
                "state_tokens": items[j]["state_stats"]["state_tokens"],
                "state_dropped": items[j]["state_stats"]["state_tokens_dropped"],
            })
        case["answers"] = answers
        case["predict"] = predicted
        cases.append(case)
        print(f"{name}: {len(answers)} question(s), {sum(len(a['ids']) for a in answers)} tokens",
              file=sys.stderr)

    tok = agent.tok
    tokenizer = [{"text": text, "ids": tok(text, add_special_tokens=False)["input_ids"]}
                 for text in TOKENIZER_STRINGS]
    ref = {
        "generator": "tools/make_laya_ref.py",
        "reference": {"laya": laya.__version__, "torch": torch.__version__,
                      "transformers": transformers.__version__},
        "fixture_sha256": {name: sha256(fixture / name) for name in (
            "model.safetensors", "tokenizer/tokenizer.json", "encoder/config.json",
            "rl_agent_config.json")},
        "tolerance": {"probs": 2e-5, "logits": 2e-4, "act_probability": 2e-4},
        "tokenizer": tokenizer,
        "cases": cases,
    }
    (fixture / "ref.json").write_text(compact_json(ref), encoding="utf-8")
    print(f"wrote {fixture / 'ref.json'}: {len(cases)} cases, {len(tokenizer)} tokenizer strings",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
