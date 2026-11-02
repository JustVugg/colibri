#!/usr/bin/env python3
"""Write gliner_decide_tiny/ref.json: what the `gliner2` package answers on the tiny fixture.

Maintainer-run, offline, once per change of the fixture: it needs torch,
transformers < 5 and the pinned `gliner2` package (pip install gliner2==2.0.0),
which CI does not. The answers come from fastino's own runtime,
AutoExtractor.from_pretrained(...).classify_text(), on the checkpoint
tools/make_gliner_decide_tiny.py writes; the SHA-256 of that checkpoint is
recorded so tests/test_gliner_decide_tiny.py refuses a regenerated fixture that
differs.

A System One request becomes one classify_text() call (decision_tasks below,
the mapping docs/gliner_decide.md states and gliner_decide.c implements): one
task per question, named by the question id, its instructions as the task's
prompt, its labels (a noul's "yes" then "no", a score's "0".."n-1"), and the
option texts as label descriptions. For every case it records what the engine
must reproduce:
  ids          the encoder's input_ids (processor._format_input_with_mapping)
  markers      the [L] positions the classifier reads (schema_special_indices)
  logits       the classifier's output per label, in the record's option order
  probs        the softmax _extract_classification_result applies, unrounded
and the package's own answer (classify_text with include_confidence), against
which the unrounded numbers are checked here before anything is written. A
long state is cut at MAX_LEN tokens the way the engine cuts it: the reference
is called with max_len set to the words that fit, its own truncation. A case
whose questions alone exceed MAX_LEN records the refusal the engine owes.
Every decision is checked for a near-tie, so a float-level difference can
never flip an argmax in the test.

    python tools/make_gliner_decide_tiny.py --output ./gliner_decide_tiny
    python tools/make_gliner_decide_ref.py --fixture ./gliner_decide_tiny
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from openai_server import systemone_decision_record  # noqa: E402

PINNED_GLINER2 = "2.0.0"
MIN_GAP = 1e-3                  # smallest top-1 / top-2 probability gap accepted
MAX_LEN = 300                   # COLI_GLINER_MAX_LEN the test runs the engine with

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
    {"role": "user", "content": "If it is not fixed today I will cancel the subscription."},
]

DEPARTMENT = {"type": "choice", "instructions": "Which department should handle this request?",
              "criteria": {"billing": "invoices, payments, refunds",
                           "technical": "bugs, outages, system errors",
                           "sales": "pricing, new contracts", "other": "everything else"}}

CASES = [
    ("ticket_three_types",
     "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan.",
     {"department": DEPARTMENT,
      "urgency": {"type": "score", "instructions": "How urgent is this request?",
                  "criteria": ["not urgent", "soon", "critical deadline or blocking issue"]},
      "churn_risk": {"type": "noul", "instructions": "Does the user threaten to cancel or leave?"},
      "refund_requested": {"type": "noul",
                           "instructions": "Does the user explicitly request a refund?"}}),
    ("two_options_and_noul_criteria",
     "The application crashes every time I open the settings page!",
     {"is_bug": {"type": "choice", "instructions": "Is this a bug report or a feature request?",
                 "criteria": {"bug": "something is broken", "feature": "something new is wanted"}},
      "blocking": {"type": "noul", "instructions": "Is the user blocked?",
                   "criteria": {"true": "the user cannot work at all",
                                "false": "the user can still work"}},
      "only_true": {"type": "noul", "instructions": "Does it mention settings?",
                    "criteria": {"true": "the settings page is named in the message"}}}),
    ("labels_without_and_with_descriptions",
     "We would like to schedule a demo of the product for our team next week",
     {"intent": {"type": "choice", "instructions": "What does the sender want?",
                 "criteria": {"demo": "a product demonstration", "refund": None, "cancel": "",
                              "pricing": {"desc": "price questions", "examples": ["cost", "plan"]},
                              "support": 3, "partnership": ["co-marketing", "reselling"],
                              "other": "anything else"}}}),
    ("twelve_labels_no_instructions",
     "I love this product, it works great and the support team was very helpful.",
     {"emotion": {"type": "choice",
                  "criteria": {name: None for name in (
                      "joy", "anger", "sadness", "fear", "surprise", "disgust", "trust",
                      "anticipation", "neutral", "confusion", "gratitude", "pride")}}}),
    ("scores",
     "This is the worst service I have ever used, nothing works and nobody answers?",
     {"sentiment": {"type": "score", "instructions": "What is the sender's tone?",
                    "criteria": ["angry", "negative", "neutral", "positive", "delighted"]},
      "six_levels": {"type": "score", "instructions": "How severe is the impact?",
                     "criteria": ["none", "minor", "moderate", "major", "critical", "catastrophic"]},
      "two_levels": {"type": "score", "instructions": "Is any action needed?",
                     "criteria": ["no action", "action needed"]},
      "numeric_levels": {"type": "score", "instructions": "Rate the urgency.",
                         "criteria": [1, 2, 3]}}),
    ("long_state_cut",
     {"subject": "Invoice problem again", "body": LONG_BODY},
     {"department": DEPARTMENT,
      "legal_threat": {"type": "noul", "instructions": "Does the sender threaten legal action?"}}),
    ("conversation", CONVERSATION,
     {"churn": {"type": "noul", "instructions": "Does the user threaten to cancel?"},
      "topic": {"type": "choice", "instructions": "What is the conversation about?",
                "criteria": {"login": "access to the account", "billing": "payments",
                             "other": "anything else"}}}),
    ("json_state_and_instructions",
     {"ticket": {"id": 4411, "amount": 129.5, "currency": "EUR", "paid": True,
                 "refund": None, "tags": ["billing", "urgent"], "nested": {"a": [1, 2.0, {"b": "c"}]}}},
     {"action": {"type": "choice",
                 "instructions": {"task": "pick the next action", "context": "support desk"},
                 "criteria": {"refund": "give the money back", "escalate": "a human decides",
                              "close": "nothing to do"}},
      "paid": {"type": "noul", "instructions": ["Was", "the", "ticket", "paid?"]}}),
    ("special_tokens_and_whitespace",
     "Order [P] arrived  broken,   see\tphoto.\r\n\r\nThe cafe [SEP_TEXT] said [L] is blocked "
     "[UNK] and support said 'we're sorry'",
     {"damaged": {"type": "noul", "instructions": "Was the item [DESCRIPTION] damaged?"},
      "channel": {"type": "choice", "instructions": "Which channel  should   answer?",
                  "criteria": {"email": "write an email [L] back",
                               "phone": "call\tthe customer", "none": None}}}),
    ("urls_mail_handles",
     "Visit https://example.com/help?x=1 or WWW.example.org, write to Help.Desk@Example.COM or "
     "ping @support_team about state-of-the-art_tools and 2026-10-02 at 3.14 pm",
     {"channel": {"type": "choice", "instructions": "Where should the user go?",
                  "criteria": {"website": None, "email": None, "social": None}}}),
    ("unicode",
     "Buongiorno, la fattura è sbagliata. Grüße aus München, ευχαριστώ ΟΔΟΣ ΣΟΦΙΑΣ. "
     "İstanbul 我们的账户被锁了 \U0001F642\U0001F44D\U0001F3FD Å Ω café",
     {"language": {"type": "choice", "instructions": "Which language comes first?",
                   "criteria": {"italian": "italiano", "german": "deutsch", "chinese": "中文"}},
      "polite": {"type": "noul", "instructions": "Is the message polite?"}}),
    ("many_options",
     "Security incident: a phishing email asked employees to reset their passwords.",
     {"category": {"type": "choice", "instructions": "Classify the security incident.",
                   "criteria": {f"cat{i:02d}": None for i in range(40)}}}),
    ("empty_state", "",
     {"anything": {"type": "noul", "instructions": "Is there any request?"}}),
    ("questions_over_max_len",
     "Too many options for the token budget.",
     {"bucket": {"type": "choice", "instructions": "Pick one.",
                 "criteria": {f"option{i}": f"the description of option {i}" for i in range(30)}}}),
]

TOKENIZER_STRINGS = [
    "", " ", "  ", "a", " a", "a ", "Hello world", "a  b\t\tc\n\nd", "tab\tand\r\nnewline\n",
    "   leading spaces", "trailing spaces   ", "x[P]y", "a [UNK] b", "[SEP_TEXT][SEP_STRUCT]",
    "[sep] [CLS] [MASK]", "<0x41>", "▁▁a", "foo▁bar", "(", ")", "[L]",
    "intent: What does the customer want? [DESCRIPTION] refund: give the money back",
    "café café Å Ω", "日本語のテキスト", "\U0001F642\U0001F44D\U0001F3FD", "a b a　　b",
    "a b", "mixed ,.;:!? punctuation!!! ... ???", "über straße Ελληνικά ΟΔΟΣ русский",
    "state-of-the-art_tools 2026-10-02 3.14",
]

SPLIT_STRINGS = [
    "Hello, World!", "state-of-the-art_tools -x- a--b _x_", "https://Example.com/a?b=1 http:// www.x",
    "HTTPS://EXAMPLE.COM wwW.Example.org www. ftp://x.y", "mail Help.Desk@Example.COM a@b.c x@y.co.uk",
    "@user @_x @ a@", "ΟΔΟΣ ΣΟΦΙΑΣ Σ ΑΣ ΣΑ", "İstanbul ıi ſ K KELVIN", "日本語 テキスト 中文",
    "café é ́", "\U0001F642 \U0001F44D\U0001F3FD", "tab\tnew\nline\x1c\x85 x",
    "2026-10-02 3.14 1,000 $5 50%", "", "   ",
]


def decision_tasks(record):
    """The classify_text() tasks for a DECIDE record: one per question, in order."""
    tasks = {}
    for question in record["questions"]:
        options = question["options"]
        if question["type"] == "noul":
            labels, texts = ["yes", "no"], [options[1]["text"], options[0]["text"]]
        else:
            labels, texts = [o["label"] for o in options], [o["text"] for o in options]
        task = {"labels": labels}
        if question["instructions"]:
            task["prompt"] = question["instructions"]
        descriptions = {label: text for label, text in zip(labels, texts) if text}
        if descriptions:
            task["label_descriptions"] = descriptions
        tasks[question["id"]] = task
    return tasks


def reference_text(state):
    """processor._collate_batch: a text that does not end in . ! or ? gets one."""
    if state and not state.endswith((".", "!", "?")):
        return state + "."
    return state or "."


class Recorder:
    """Hooks on the reference model: the encoder's input ids, the classifier's
    logits per task, the marker positions the processor routed."""

    def __init__(self, model):
        import torch
        self.ids, self.logits, self.markers = [], [], []
        model.encoder.register_forward_pre_hook(
            lambda module, args, kwargs: self.ids.append(kwargs["input_ids"][0].tolist()),
            with_kwargs=True)
        model.classifier.register_forward_hook(
            lambda module, args, out: self.logits.append(out.squeeze(-1).to(torch.float64).tolist()))
        processor = model.processor
        original = processor.extract_embeddings_from_batch

        def routed(token_embeddings, input_ids, batch):
            self.markers.append(batch.schema_special_indices[0])
            return original(token_embeddings, input_ids, batch)
        processor.extract_embeddings_from_batch = routed

    def clear(self):
        self.ids.clear(); self.logits.clear(); self.markers.clear()


def fit_words(model, record, tasks, max_len):
    """How many words of the state the engine keeps under max_len tokens, found
    with the reference's own splitter and tokenizer: (schema tokens, words,
    tokens per word)."""
    processor = model.processor
    schema = model._classification_schema(tasks).build()
    schema = {**schema, "classifications": [{**c, "true_label": ["N/A"]} for c in schema["classifications"]]}
    processor.change_mode(is_training=False)
    schema_tokens = len(processor.transform_and_format("", schema).input_ids)   # up to [SEP_TEXT]
    words = [w for w, _, _ in processor.word_splitter(reference_text(record["state"]), lower=True)]
    counts = [len(processor.tokenizer.tokenize(w)) for w in words]
    return schema_tokens, words, counts


def run_case(model, recorder, record, max_len, min_gap=MIN_GAP):
    """One record through the reference: what the engine must reproduce, or the
    refusal it owes. min_gap=None accepts near-ties (the comparison tool)."""
    import numpy as np
    tasks = decision_tasks(record)
    schema_tokens, words, counts = fit_words(model, record, tasks, max_len)
    if schema_tokens > max_len:
        return {"refused": f"questions: the questions and their options take {schema_tokens} tokens, "
                           f"more than max_len={max_len}"}
    used, total = 0, schema_tokens
    for count in counts:
        if total + count > max_len:
            break
        total += count
        used += 1
    recorder.clear()
    result = model.classify_text(record["state"], tasks, include_confidence=True,
                                 max_len=None if used == len(words) else used)
    if len(recorder.ids) != 1 or len(recorder.logits) != len(tasks) or len(recorder.markers) != 1:
        sys.exit("the reference ran more than one forward or skipped a task")
    markers = []
    for positions in recorder.markers[0]:
        markers.extend(positions[1:])
    answers = []
    for question, logits in zip(record["questions"], recorder.logits):
        z = np.array(logits, dtype=np.float64)
        p = np.exp(z - z.max())
        p = p / p.sum()
        order = np.sort(p)[::-1]
        if min_gap is not None and len(p) > 1 and order[0] - order[1] < min_gap:
            sys.exit(f"{question['id']}: top-2 gap {order[0] - order[1]:.2e} is a near-tie; "
                     "change the case or SEED")
        got = result[question["id"]]
        labels = tasks[question["id"]]["labels"]
        best = int(np.argmax(p))
        if got["label"] != labels[best] or abs(got["confidence"] - float(p[best])) > 1e-6:
            sys.exit(f"{question['id']}: unrounded probabilities disagree with classify_text "
                     f"({got} vs {labels[best]} {p[best]})")
        if question["type"] == "noul":                   # read as yes, no; recorded false, true
            z, p = z[::-1], p[::-1]
        answers.append({"id": question["id"], "logits": [float(x) for x in z],
                        "probs": [float(x) for x in p]})
    return {"ids": recorder.ids[0], "markers": markers, "words": len(words), "words_used": used,
            "state_tokens": sum(counts), "state_used": sum(counts[:used]),
            "answers": answers, "predict": result}


def compact_json(ref):
    """One key per line at the top, one case or string per line below it:
    diffable, without a line per token id."""
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


def load_reference(path, allow_version=False):
    import importlib.metadata
    import torch
    version = importlib.metadata.version("gliner2")
    if version != PINNED_GLINER2 and not allow_version:
        sys.exit(f"gliner2 {version} is installed; the reference is pinned to {PINNED_GLINER2}")
    from gliner2 import AutoExtractor
    torch.manual_seed(0)
    model = AutoExtractor.from_pretrained(str(path))
    model.eval()
    return model, version


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fixture", default="./gliner_decide_tiny")
    parser.add_argument("--allow-version", action="store_true",
                        help=f"run with an installed gliner2 other than {PINNED_GLINER2}")
    args = parser.parse_args()

    import torch
    import transformers
    from gliner2.processing.word_splitter import WhitespaceTokenSplitter

    fixture = Path(args.fixture)
    model, version = load_reference(fixture, args.allow_version)
    recorder = Recorder(model)
    cases = []
    for name, state, questions in CASES:
        record = systemone_decision_record({"state": state, "questions": questions})
        case = {"name": name, "state": state, "questions": questions}
        case.update(run_case(model, recorder, record, MAX_LEN))
        cases.append(case)
        what = case.get("refused") or f"{len(case['answers'])} question(s), {len(case['ids'])} tokens"
        print(f"{name}: {what}", file=sys.stderr)

    tok = model.processor.tokenizer
    tokenizer = [{"text": text, "ids": tok.convert_tokens_to_ids(tok.tokenize(text))}
                 for text in TOKENIZER_STRINGS]
    splitter = WhitespaceTokenSplitter()
    split = [{"text": text, "words": [w for w, _, _ in splitter(text, lower=True)]}
             for text in SPLIT_STRINGS]
    ref = {
        "generator": "tools/make_gliner_decide_ref.py",
        "reference": {"gliner2": version, "torch": torch.__version__,
                      "transformers": transformers.__version__},
        "fixture_sha256": {name: sha256(fixture / name) for name in (
            "model.safetensors", "tokenizer.json", "config.json", "encoder_config/config.json")},
        "max_len": MAX_LEN,
        "tolerance": {"probs": 2e-5, "logits": 1e-4},
        "tokenizer": tokenizer,
        "split": split,
        "cases": cases,
    }
    (fixture / "ref.json").write_text(compact_json(ref), encoding="utf-8")
    print(f"wrote {fixture / 'ref.json'}: {len(cases)} cases, {len(tokenizer)} tokenizer strings, "
          f"{len(split)} split strings", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
