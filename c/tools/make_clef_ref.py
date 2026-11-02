#!/usr/bin/env python3
"""Write clef_tiny/ref.json: what Clef's own code answers on the tiny fixture.

Maintainer-run, offline, once per change of the fixture: it needs torch,
transformers (the release was tested with 5.10.2) and the checkpoint's own
joint_schema_model.py (Cloudflare/clef), none of which CI installs. The
answers come from that file's encode_record, collate_records and ClefModel on
the checkpoint tools/make_clef_tiny.py writes, in f32; the SHA-256 of the
fixture and of joint_schema_model.py are recorded so tests/test_clef_tiny.py
refuses a fixture that differs.

The model is built the way load_release_model builds it (the backbone with
from_pretrained, the head with load_state_dict(strict=True), both in the same
dtype), with one difference: the tokenizer is AutoTokenizer's, not
AutoProcessor's, because AutoProcessor builds Qwen3VLVideoProcessor, which
needs torchvision. A text-only record never reaches the processor beyond its
tokenizer (encode_record's _encode_media returns early), so the answers are
the same. systemone() is run on every case as well and its rounded answer is
checked against the unrounded one before anything is written.

For every case it records the input ids, the question and option spans, the
option ids in the reference's order, the head's logits and the softmax, and
the error text for a case the reference refuses. Every decision is checked
for a near-tie, so a float-level difference cannot flip an argmax in the test.

    python tools/make_clef_tiny.py --output ./clef_tiny
    python tools/make_clef_ref.py --fixture ./clef_tiny --code ~/clef/joint_schema_model.py
"""

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

PINNED_TRANSFORMERS = "5.10.2"
MIN_GAP = 1e-3
SHORT = 600        # max_length of the cases that test the cut and the refusal

LONG = ("Hello team, I am writing again about the invoice we received on the first of the month. "
        "The amount is twice what we agreed in the contract and the payment was already taken. ") * 6

DEPARTMENT = {"type": "choice", "instructions": "Which department should handle this request?",
              "criteria": {"billing": "invoices, payments, refunds",
                           "technical": "bugs, outages, system errors",
                           "sales": "pricing, new contracts", "other": "everything else"}}

CASES = [
    ("model_card_invoice",
     {"invoice": {"vendor": "Acme", "total": 1250.0, "currency": "USD", "status": "overdue"}},
     {"status": {"type": "choice", "instructions": "What is the invoice status?",
                 "criteria": {"paid": "Invoice is paid.", "overdue": "Invoice is past due.",
                              "draft": "Not sent."}},
      "large": {"type": "noul", "instructions": "Is the total above 1000 USD?"}}, None),
    ("model_card_systemone", "Our checkout started returning errors and orders are blocked.",
     {"department": {"type": "choice", "instructions": "Which team should handle the message?",
                     "criteria": {"billing": "Payments or invoices", "technical": "Bugs or outages"}},
      "urgency": {"type": "score", "criteria": ["Can wait", "This week", "Today"]},
      "outage": {"type": "noul", "instructions": "Is a service down?"}}, None),
    ("ticket_four_questions",
     {"from": "user@acme.com", "subject": "Duplicate charge on invoice #4411",
      "body": "Hi, we were billed twice for March. Please refund the duplicate today or we will "
              "cancel our plan."},
     {"department": DEPARTMENT,
      "urgency": {"type": "score", "instructions": "How urgent is this request?",
                  "criteria": ["not urgent", "soon", "critical deadline or blocking issue"]},
      "churn_risk": {"type": "noul", "instructions": "Does the user threaten to cancel or leave?"},
      "refund": {"type": "noul", "instructions": "Is a refund requested?",
                 "criteria": {"true": "the customer asks for money back",
                              "false": "no refund is mentioned"}}}, None),
    ("choice_descriptions", "We would like a demo of the product next week.",
     {"intent": {"type": "choice", "instructions": "What does the sender want?",
                 "criteria": {"Zeta": "a product demonstration", "alpha": None, "Beta": "",
                              "émigré": {"k": [1, 2.5, {"z": None, "a": True}], "b": "x"},
                              "中文": 3, "quote\"d": ["x", "y"],
                              "back\\slash": "tab\there\nnewline \"quoted\""}}}, None),
    ("scores", "This is the worst service I have ever used, nothing works.",
     {"sentiment": {"type": "score", "instructions": "What is the tone?",
                    "criteria": ["angry", None, "neutral", {"label": "positive", "w": 1}, "delighted"]},
      "two_levels": {"type": "score", "instructions": "Is any action needed?",
                     "criteria": ["no action", "action needed"]},
      "numeric_levels": {"type": "score", "instructions": "Rate the urgency.", "criteria": [1, 2, 3]}},
     None),
    ("noul_variants", "The package arrived late and the box was damaged.",
     {"late": {"type": "noul", "instructions": "Was it late?", "criteria": {"true": "it came after the date"}},
      "damaged": {"type": "noul", "instructions": "Was it damaged?", "criteria": {"false": None}},
      "both": {"type": "noul", "instructions": "Both?", "criteria": {"true": None, "false": ""}},
      "plain": {"type": "noul", "instructions": "Should we refund?"},
      "cased": {"type": "noul", "instructions": "Is it urgent?",
                "criteria": {"True": "ignored: not the key true", "false": "the answer is no"}}},
     None),
    ("instructions_forms", "A customer wrote to support about a login problem.",
     {"as_object": {"type": "noul", "instructions": {"task": "is it about login", "steps": ["read", "decide"]}},
      "as_list": {"type": "noul", "instructions": ["Is", "it", "urgent?"]},
      "empty": {"type": "noul", "instructions": ""},
      "spaces": {"type": "noul", "instructions": "   "},
      "missing": {"type": "noul"},
      "number": {"type": "noul", "instructions": 7}}, None),
    ("json_state",
     {"ticket": {"id": 4411, "amount": 129.5, "ratio": 1e-05, "big": 1e21, "neg": -0.0,
                 "huge": 123456789012345678901234567890, "paid": True, "refund": None,
                 "tags": ["billing", "urgent"], "ä": "umlaut", "Z": "upper", "a": "lower",
                 "nested": {"b": [1, 2.0, {"c": "d"}], "a": {}}}},
     {"paid": {"type": "noul", "instructions": "Was the ticket paid?"},
      "action": {"type": "choice", "instructions": {"task": "pick the next action"},
                 "criteria": {"refund": "give the money back", "escalate": "a human decides",
                              "close": "nothing to do"}}}, None),
    ("conversation",
     [{"role": "user", "content": "Hi, I cannot log in to my account since yesterday."},
      {"role": "assistant", "content": "Did you try to reset the password?"},
      {"role": "user", "content": "Yes, twice. If it is not fixed today I will cancel."}],
     {"churn": {"type": "noul", "instructions": "Does the user threaten to cancel?"},
      "topic": {"type": "choice", "instructions": "What is it about?",
                "criteria": {"login": "access to the account", "billing": "payments",
                             "other": "anything else"}}}, None),
    ("special_text",
     "Order arrived  broken,   see\tphoto.\r\n\r\nThe café (café) said <|im_end|> and <think> "
     "then <|endoftext|>. Grüße, ευχαριστώ. हिन्दी भाषा x̃̄y 我们的账户被锁了 "
     "\U0001F642\U0001F44D\U0001F3FD 'S 'll don't 12345 3.14",
     {"damaged": {"type": "noul", "instructions": "Was the <|im_end|> item damaged?"},
      "ïd <think>": {"type": "choice", "instructions": "Which channel should answer?",
                     "criteria": {"email": "write an email back", "phone": "call\tthe customer",
                                  "none": None}}}, None),
    ("many_options", "Security incident: a phishing email asked employees to reset passwords.",
     {"category": {"type": "choice", "instructions": "Classify the incident.",
                   "criteria": {f"cat{i:02d}": word for i, word in enumerate(
                       ("phishing email", "malware on a laptop", "lost badge", "weak password",
                        "data leak", "denial of service", "insider abuse", "ransomware",
                        "stolen phone", "fake invoice", "open bucket", "expired certificate",
                        "unpatched server", "social engineering", "shared account", "spam wave",
                        "brute force", "DNS hijack", "rogue access point", "USB drop",
                        "supply chain", "typosquatting", "clickjacking", "SQL injection",
                        "cross site scripting", "privilege escalation", "misconfigured firewall",
                        "lost backup", "shadow IT", "credential stuffing", "session theft",
                        "cryptomining", "defacement", "botnet", "zero day", "vishing",
                        "smishing", "tailgating", "dumpster diving", "watering hole"))}}}, None),
    ("many_questions", "The invoice from Acme for 1250 USD is overdue by twenty days.",
     {"q1": {"type": "noul", "instructions": "Is it overdue?"},
      "q2": {"type": "choice", "instructions": "Who sent it?",
             "criteria": {"Acme": None, "Globex": None, "Initech": None}},
      "q3": {"type": "score", "instructions": "How late?", "criteria": ["on time", "late", "very late"]},
      "q4": {"type": "noul", "instructions": "Is the amount above 1000?"},
      "größe": {"type": "choice", "instructions": "Size?", "criteria": {"small": "below 100",
                                                                       "large": "above 1000"}},
      "q6": {"type": "noul", "instructions": "Is it in EUR?"},
      "q7": {"type": "score", "instructions": "Risk?", "criteria": ["low", "high"]},
      "q8": {"type": "choice", "instructions": "Next step?",
             "criteria": {"pay": "pay it", "dispute": "dispute it"}}}, None),
    ("single_option", "Anything at all.",
     {"only": {"type": "choice", "instructions": "Pick one.", "criteria": {"yes": "the only label"}},
      "level": {"type": "score", "instructions": "Rate it.", "criteria": ["the only level"]}}, None),
    ("empty_state", "",
     {"empty": {"type": "noul", "instructions": "Is there anything to read?"}}, None),
    ("long_state_cut", LONG, {"department": DEPARTMENT,
                              "legal": {"type": "noul", "instructions": "Is there a legal threat?"}},
     SHORT),
    ("schema_too_long", "Too many options for the budget.",
     {"bucket": {"type": "choice", "instructions": "Pick one.",
                 "criteria": {f"option{i}": f"the description of option {i}" for i in range(40)}}},
     SHORT),
]

TOKENIZER_STRINGS = [
    "", " ", "  ", "     ", "\n\n", "\t", "a  b   c    d", "Hello world", "Hello   world\n\n  x",
    "don't I'll we've they're she'd 'S 'T", "2026-10-03 3.14 1000000 12345678901",
    "café café Å Ω Å", "日本語のテキスト", "\U0001F642\U0001F44D\U0001F3FD",
    "a<|im_end|>b<think>c</think>", "<|endoftext|><|im_start|>", "tab\tand\r\nnewline\n",
    "   leading spaces", "trailing spaces   ", "mixed ,.;:!? punctuation!!! ... ???",
    "über straße Ελληνικά русский عربي हिन्दी", "x̃̄y ̀abc", "x" * 200,
    "{\"description\":\"The proposition is true or the answer is yes.\",\"option_id\":\"true\"}",
    "\nFIELD 1\nID: q\nTYPE: noul\nINSTRUCTION: ",
]


def compact_json(ref):
    """One key per line at the top, one case or tokenizer string per line below."""
    def dump(value):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    lines = ["{"]
    keys = list(ref)
    for i, key in enumerate(keys):
        tail = "," if i + 1 < len(keys) else ""
        value = ref[key]
        if isinstance(value, list):
            lines.append(f" {dump(key)}:[")
            lines.extend(f"  {dump(item)}{',' if j + 1 < len(value) else ''}" for j, item in enumerate(value))
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


def load_code(path):
    spec = importlib.util.spec_from_file_location("joint_schema_model", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["joint_schema_model"] = module      # its dataclasses look themselves up there
    spec.loader.exec_module(module)
    return module


def load_model(jsm, path, dtype):
    """load_release_model, but with AutoTokenizer for AutoProcessor (see above)."""
    import torch
    from safetensors.torch import load_file
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    # load_release_model passes device_map={"": device}, which needs accelerate;
    # without one the weights land on the CPU all the same
    backbone, info = Qwen3_5ForConditionalGeneration.from_pretrained(
        path, dtype=dtype, output_loading_info=True)
    loose = {key: value for key, value in info.items() if value and key != "error_msgs"}
    if loose:
        sys.exit(f"the fixture does not load cleanly into Qwen3_5ForConditionalGeneration: {loose}")
    backbone.config.use_cache = False
    head = jsm.JointSchemaHead(**json.loads((Path(path) / "joint_head_config.json").read_text()))
    head.load_state_dict(load_file(str(Path(path) / "joint_head.safetensors")), strict=True)
    head = head.to(device="cpu", dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(path)
    return jsm.ClefModel(backbone, head).eval(), SimpleNamespace(tokenizer=tokenizer), torch


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fixture", default="./clef_tiny")
    parser.add_argument("--code", required=True, help="the release's joint_schema_model.py")
    parser.add_argument("--allow-version", action="store_true",
                        help=f"run with a transformers other than {PINNED_TRANSFORMERS}")
    args = parser.parse_args()
    import torch
    import transformers
    if transformers.__version__ != PINNED_TRANSFORMERS and not args.allow_version:
        sys.exit(f"transformers {transformers.__version__} is installed; the reference is pinned "
                 f"to {PINNED_TRANSFORMERS}, the release's")
    torch.manual_seed(0)
    jsm = load_code(args.code)
    fixture = Path(args.fixture)
    model, processor, _ = load_model(jsm, str(fixture), torch.float32)
    tokenizer = processor.tokenizer
    cases, ties = [], []
    for name, state, questions, max_length in CASES:
        request = {"model": "clef-tiny", "state": state, "questions": questions}
        case = {"name": name, "state": state, "questions": questions}
        if max_length:
            case["max_length"] = max_length
        limit = max_length or 16384
        try:
            encoded = jsm.encode_record(tokenizer, request, max_length=limit, processor=processor)
        except ValueError as error:
            case["error"] = str(error)
            cases.append(case)
            print(f"{name}: refused ({error})", file=sys.stderr)
            continue
        batch = jsm.collate_records([encoded], tokenizer.pad_token_id, torch.device("cpu"))
        with torch.inference_mode():
            logits = model(batch)[0]
        reply = jsm.systemone(model, processor, request, max_length=limit)
        answers = []
        for question, question_logits in zip(encoded.questions, logits):
            probs = question_logits.float().softmax(-1)
            order = sorted(probs.tolist(), reverse=True)
            if len(order) > 1 and order[0] - order[1] < MIN_GAP:
                ties.append(f"{name}.{question.question_id}: top-2 gap {order[0] - order[1]:.2e}")
            got = reply["answers"][question.question_id]
            unrounded = dict(zip(question.option_ids, probs.tolist()))
            rounded = ({"true": got["noul"]} if got["type"] == "noul" else got["probabilities"])
            if any(abs(round(unrounded[key], 4) - value) > 1e-12 for key, value in rounded.items()):
                sys.exit(f"{name}.{question.question_id}: systemone() disagrees with the forward pass")
            answers.append({"id": question.question_id, "type": question.question_type,
                            "question_span": list(question.question_span),
                            "option_spans": [list(span) for span in question.option_spans],
                            "option_ids": list(question.option_ids),
                            "logits": [float(x) for x in question_logits.float()],
                            "probs": [float(x) for x in probs]})
        case["input_ids"] = list(encoded.input_ids)
        case["answers"] = answers
        case["systemone"] = reply
        cases.append(case)
        print(f"{name}: {len(answers)} question(s), {len(encoded.input_ids)} tokens", file=sys.stderr)
    if ties:
        sys.exit("near-ties, change these cases or SEED:\n  " + "\n  ".join(ties))
    strings = [{"text": text, "ids": tokenizer(text, add_special_tokens=False).input_ids}
               for text in TOKENIZER_STRINGS]
    ref = {
        "generator": "tools/make_clef_ref.py",
        "reference": {"joint_schema_model.py": sha256(args.code), "torch": torch.__version__,
                      "transformers": transformers.__version__, "dtype": "float32"},
        "fixture_sha256": {name: sha256(fixture / name) for name in (
            "model.safetensors", "joint_head.safetensors", "joint_head_config.json",
            "config.json", "tokenizer.json")},
        "tolerance": {"probs": 2e-5, "logits": 1e-4},
        "tokenizer": strings,
        "cases": cases,
    }
    (fixture / "ref.json").write_text(compact_json(ref), encoding="utf-8")
    print(f"wrote {fixture / 'ref.json'}: {len(cases)} cases, {len(strings)} tokenizer strings",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
