#!/usr/bin/env python3
"""qwen36's Clef path against Clef's own code on the real checkpoint (maintainer tool).

Runs the same requests through both and reports, per request and overall, the
largest probability difference, whether every decision (argmax) agrees, and the
engine's time per request. The requests are the model card's examples
(Cloudflare/clef README: the invoice record and the systemone() call), the
Jev / System One examples of docs/systemone.md and docs/laya.md, and cases of
our own: a contract read in full, a conversation, unicode, JSON states and
instructions, a 20-intent choice, nine questions in one record.

Two halves, usually on the same box at different times (a 27B in bf16 and the
engine do not fit in RAM together):

  reference   Clef's joint_schema_model.py (the copy in the checkpoint) on the
              Hugging Face checkpoint, in bf16 as the release runs, through
              transformers. The backbone's later layers can be offloaded to disk
              (--cpu-layers) so it fits beside nothing else; the answers are the
              same, only slower. Written to --reference and read back from it.
  engine      ./qwen36 on the converted container (tools/convert_qwen36.py),
              CLEF_RECORDS mode: the records the gateway sends in the raw form,
              one process, every request in turn.

    python tools/compare_clef.py --hf ~/clef --reference clef_ref.json --cpu-layers 36
    python tools/compare_clef.py --container ~/clef_c --reference clef_ref.json \\
        [--env COLI_DENSE_BITS=4] [--json out.json]
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from openai_server import systemone_decision_record  # noqa: E402

CONTRACT = (
    "MASTER SERVICES AGREEMENT. This agreement is made between the Provider and the Customer. "
    "1. Services. The Provider will deliver the hosted platform described in Schedule A, including "
    "updates, security patches and support during business hours. 2. Fees. The Customer pays the "
    "fees in Schedule B within thirty days of each invoice; late amounts accrue interest at one "
    "percent per month. 3. Term. The initial term is twelve months and renews automatically for "
    "successive twelve-month periods unless either party gives sixty days written notice. "
    "4. Termination. Either party may terminate for material breach not cured within thirty days "
    "of notice. 5. Confidentiality. Each party keeps the other's confidential information secret "
    "and uses it only to perform this agreement. 6. Data protection. The Provider processes "
    "personal data only on documented instructions and notifies the Customer of any breach within "
    "seventy-two hours. 7. Liability. Neither party is liable for indirect damages; total liability "
    "is capped at the fees paid in the twelve months before the claim. 8. Governing law. This "
    "agreement is governed by the laws of the State of New York. ")
BANKING = ["activate_my_card", "apple_pay", "card_arrival", "card_not_working", "change_pin",
           "compromised_card", "declined_card_payment", "exchange_rate", "failed_transfer",
           "lost_or_stolen_card", "pending_transfer", "refund_not_showing_up", "top_up_failed",
           "transfer_timing", "verify_my_identity", "wrong_amount_of_cash_received",
           "cash_withdrawal_charge", "direct_debit_payment_not_recognised", "terminate_account",
           "age_limit"]


def records():
    return [
        ("card invoice (encode_record example)",
         {"invoice": {"vendor": "Acme", "total": 1250.0, "currency": "USD", "status": "overdue"}},
         {"status": {"type": "choice", "instructions": "What is the invoice status?",
                     "criteria": {"paid": "Invoice is paid.", "overdue": "Invoice is past due.",
                                  "draft": "Not sent."}},
          "large": {"type": "noul", "instructions": "Is the total above 1000 USD?"}}),
        ("card systemone example", "Our checkout started returning errors and orders are blocked.",
         {"department": {"type": "choice", "instructions": "Which team should handle the message?",
                         "criteria": {"billing": "Payments or invoices", "technical": "Bugs or outages"}},
          "urgency": {"type": "score", "criteria": ["Can wait", "This week", "Today"]},
          "outage": {"type": "noul", "instructions": "Is a service down?"}}),
        ("systemone.md curl",
         "Hi, I have been trying to connect my Stripe account for 3 days and the integration keeps "
         "failing. I am losing sales. Please help ASAP.",
         {"urgency": {"type": "noul", "instructions": "Does this message express urgency?"},
          "department": {"type": "choice", "instructions": "Which team should handle this?",
                         "criteria": {"billing": "payments, invoices, Stripe payouts",
                                      "technical": "bugs, outages, integration errors",
                                      "sales": "pricing, plans, upgrades"}},
          "severity": {"type": "score", "instructions": "How severe is the customer impact?",
                       "criteria": ["no impact", "minor inconvenience", "blocked on one task",
                                    "losing money", "business down"]}}),
        ("laya.md readme",
         "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel "
         "our plan.",
         {"department": {"type": "choice", "instructions": "Which department should handle this?",
                         "criteria": {"billing": "invoices, payments, refunds",
                                      "technical": "bugs, outages, system errors",
                                      "other": "everything else"}},
          "urgency": {"type": "score", "instructions": "How urgent is this?",
                      "criteria": ["not urgent", "soon", "blocking"]},
          "churn_risk": {"type": "noul", "instructions": "Does the user threaten to cancel or leave?"}}),
        ("ticket json + refund criteria",
         {"from": "user@acme.com", "subject": "Duplicate charge on invoice #4411",
          "body": "Hi, we were billed twice for March. Please refund the duplicate today or we will "
                  "cancel our plan."},
         {"refund": {"type": "noul", "instructions": "Is a refund requested?",
                     "criteria": {"true": "the customer asks for money back",
                                  "false": "no refund is mentioned"}},
          "sentiment": {"type": "score", "instructions": "What is the sender's tone?",
                        "criteria": ["angry", "negative", "neutral", "positive", "delighted"]}}),
        ("contract in full", CONTRACT,
         {"renewal": {"type": "noul", "instructions": "Does the agreement renew automatically?"},
          "notice": {"type": "choice", "instructions": "How much notice stops the renewal?",
                     "criteria": {"thirty days": None, "sixty days": None, "ninety days": None}},
          "law": {"type": "choice", "instructions": "Which law governs the agreement?",
                  "criteria": {"New York": None, "California": None, "Delaware": None,
                               "England and Wales": None}},
          "breach_hours": {"type": "score", "instructions": "How fast must a data breach be reported?",
                           "criteria": ["within 24 hours", "within 72 hours", "within 30 days"]}}),
        ("code review", "The PR touches the engine and carries no tests. CI is red on two jobs.",
         {"verdict": {"type": "choice", "instructions": "What should we do with the pull request?",
                      "criteria": {"merge": None, "request changes": None, "close": None}}}),
        ("conversation churn",
         [{"role": "user", "content": "Hi, I cannot log in to my account since yesterday."},
          {"role": "assistant", "content": "Sorry about that. Did you try to reset the password?"},
          {"role": "user", "content": "Yes, twice, and the reset email never arrives."},
          {"role": "user", "content": "If it is not fixed today I will cancel the subscription."}],
         {"churn": {"type": "noul", "instructions": "Does the user threaten to cancel?"},
          "topic": {"type": "choice", "instructions": "What is the conversation about?",
                    "criteria": {"login": "access to the account", "billing": "payments",
                                 "other": "anything else"}}}),
        ("unicode italian", "Buongiorno, la fattura di marzo è sbagliata: ci avete addebitato due "
         "volte. Grazie.",
         {"language": {"type": "choice", "instructions": "Which language is the message in?",
                       "criteria": {"italian": "italiano", "german": "deutsch", "english": "english"}},
          "billing": {"type": "noul", "instructions": "Is it about billing?"}}),
        ("unicode mixed", "我们的账户被锁了，请尽快处理。 Grüße aus München \U0001F642",
         {"locked": {"type": "noul", "instructions": "Is an account locked?"},
          "polite": {"type": "noul", "instructions": "Is the message polite?"}}),
        ("json instructions", {"ticket": {"id": 4411, "amount": 129.5, "currency": "EUR", "paid": True}},
         {"action": {"type": "choice",
                     "instructions": {"task": "pick the next action", "context": "support desk"},
                     "criteria": {"refund": "give the money back", "escalate": "a human decides",
                                  "close": "nothing to do"}},
          "paid": {"type": "noul", "instructions": ["Was", "the", "ticket", "paid?"]}}),
        ("banking 20 intents", "I tried to pay with my card at the supermarket and it was refused "
         "twice, although there is money on the account.",
         {"intent": {"type": "choice", "instructions": "Which intent does the customer express?",
                     "criteria": {name: name.replace("_", " ") for name in BANKING}}}),
        ("capital (mmlu-like)", "Question: What is the capital of Australia?",
         {"answer": {"type": "choice", "instructions": "Pick the correct answer.",
                     "criteria": {"A": "Sydney", "B": "Canberra", "C": "Melbourne", "D": "Perth"}}}),
        ("arithmetic", "A train travels 120 km in 2 hours.",
         {"speed": {"type": "choice", "instructions": "What is its average speed?",
                    "criteria": {"40 km/h": None, "60 km/h": None, "80 km/h": None, "240 km/h": None}},
          "faster_than_100": {"type": "noul", "instructions": "Is it faster than 100 km/h?"}}),
        ("nine questions", "Invoice INV-2207 from Globex: 3,400 EUR for consulting in August, due "
         "September 30. Approved by Maria. Paid on October 2 by bank transfer.",
         {"vendor": {"type": "choice", "criteria": {"Acme": None, "Globex": None, "Initech": None}},
          "currency": {"type": "choice", "criteria": {"USD": None, "EUR": None, "GBP": None}},
          "paid": {"type": "noul", "instructions": "Has it been paid?"},
          "late": {"type": "noul", "instructions": "Was it paid after the due date?"},
          "approved": {"type": "noul", "instructions": "Was it approved?"},
          "method": {"type": "choice", "instructions": "How was it paid?",
                     "criteria": {"card": None, "bank transfer": None, "cash": None}},
          "amount": {"type": "score", "instructions": "How large is the amount?",
                     "criteria": ["below 1,000", "1,000 to 5,000", "above 5,000"]},
          "service": {"type": "choice", "instructions": "What was bought?",
                      "criteria": {"consulting": None, "hardware": None, "software": None}},
          "month": {"type": "choice", "instructions": "Which month of service?",
                    "criteria": {"July": None, "August": None, "September": None}}}),
    ]


def reference_answers(args, cases):
    """Clef's own code, unrounded, in bf16 as the release runs."""
    import importlib.util
    import torch
    from safetensors.torch import load_file
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    if args.threads:
        torch.set_num_threads(args.threads)
    spec = importlib.util.spec_from_file_location("joint_schema_model", Path(args.hf) / "joint_schema_model.py")
    jsm = importlib.util.module_from_spec(spec)
    sys.modules["joint_schema_model"] = jsm
    spec.loader.exec_module(jsm)
    dtype = getattr(torch, args.dtype)
    kwargs = {}
    if args.cpu_layers:
        # load_release_model with device_map={"": "cpu"}, except that the layers past
        # --cpu-layers (and the vision tower, which a text record never runs) stay on
        # disk and are read in at each forward: the same weights, less RAM
        config = json.loads((Path(args.hf) / "config.json").read_text())
        layers = config["text_config"]["num_hidden_layers"]
        device_map = {"model.visual": "disk", "lm_head": "cpu",
                      "model.language_model.embed_tokens": "cpu",
                      "model.language_model.norm": "cpu", "model.language_model.rotary_emb": "cpu"}
        for i in range(layers):
            device_map[f"model.language_model.layers.{i}"] = "cpu" if i < args.cpu_layers else "disk"
        offload = Path(tempfile.mkdtemp(prefix="clef-offload-"))
        kwargs = {"device_map": device_map, "offload_folder": str(offload)}
    backbone = Qwen3_5ForConditionalGeneration.from_pretrained(args.hf, dtype=dtype, **kwargs)
    backbone.config.use_cache = False
    head = jsm.JointSchemaHead(**json.loads((Path(args.hf) / "joint_head_config.json").read_text()))
    head.load_state_dict(load_file(str(Path(args.hf) / "joint_head.safetensors")), strict=True)
    head = head.to(device="cpu", dtype=dtype)
    model = jsm.ClefModel(backbone, head).eval()
    # AutoProcessor builds the video processor, which needs torchvision; a text-only
    # record reaches nothing of the processor but its tokenizer
    processor = SimpleNamespace(tokenizer=AutoTokenizer.from_pretrained(args.hf))
    out = []
    for name, state, questions in cases:
        request = {"model": "clef", "state": state, "questions": questions}
        started = time.perf_counter()
        encoded = jsm.encode_record(processor.tokenizer, request, processor=processor)
        batch = jsm.collate_records([encoded], processor.tokenizer.pad_token_id, torch.device("cpu"))
        with torch.inference_mode():
            logits = model(batch)[0]
        seconds = time.perf_counter() - started
        answers = [{"id": q.question_id, "option_ids": list(q.option_ids),
                    "logits": [float(x) for x in l.float()],
                    "probs": [float(x) for x in l.float().softmax(-1)]}
                   for q, l in zip(encoded.questions, logits)]
        out.append({"name": name, "input_ids": list(encoded.input_ids), "answers": answers,
                    "seconds": seconds})
        print(f"reference {name}: {len(encoded.input_ids)} tokens, {seconds:.1f} s", flush=True)
    return {"dtype": args.dtype, "cases": out}


def engine_answers(args, cases):
    payloads = [{"payload": json.dumps(systemone_decision_record({"state": s, "questions": q}, "raw"),
                                       ensure_ascii=False)} for _, s, q in cases]
    env = dict(os.environ, SNAP=args.container, CLEF_IDS="1")
    for item in args.env:
        key, _, value = item.partition("=")
        env[key] = value
    if args.threads:
        env["OMP_NUM_THREADS"] = str(args.threads)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "records.json"
        path.write_text(json.dumps(payloads, ensure_ascii=False), encoding="utf-8")
        env["CLEF_RECORDS"] = str(path)
        run = subprocess.run([args.engine, "8", "8"], capture_output=True, text=True,
                             encoding="utf-8", env=env)
    if run.returncode:
        sys.exit(run.stderr[-4000:])
    return [json.loads(line) for line in run.stdout.splitlines()]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hf", help="the Hugging Face checkpoint (for the reference)")
    parser.add_argument("--container", help="the converted container (for the engine)")
    parser.add_argument("--engine", default=str(HERE / "qwen36"))
    parser.add_argument("--reference", required=True, help="the reference's answers: read from "
                        "this file when it exists, else computed (--hf) and written to it")
    parser.add_argument("--dtype", default="bfloat16", help="the reference's dtype (the release: bfloat16)")
    parser.add_argument("--cpu-layers", type=int, default=0,
                        help="reference: keep this many backbone layers in RAM, the rest on disk")
    parser.add_argument("--env", action="append", default=[], help="KEY=VALUE for the engine")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--json", help="write the per-request comparison here")
    args = parser.parse_args()
    cases = records()
    path = Path(args.reference)
    if path.exists():
        reference = json.loads(path.read_text(encoding="utf-8"))
    else:
        if not args.hf:
            sys.exit("no reference file yet: pass --hf to compute it")
        reference = reference_answers(args, cases)
        path.write_text(json.dumps(reference, ensure_ascii=False) + "\n", encoding="utf-8")
    if not args.container:
        return 0
    started = time.perf_counter()
    got = engine_answers(args, cases)
    wall = time.perf_counter() - started
    results, worst, agree, total, ids_same = [], 0.0, 0, 0, 0
    for (name, state, questions), mine, ref in zip(cases, got, reference["cases"]):
        if "error" in mine:
            sys.exit(f"{name}: the engine refused: {mine['error']}")
        same_ids = mine["input_ids"] == ref["input_ids"]
        ids_same += same_ids
        record = systemone_decision_record({"state": state, "questions": questions}, "raw")
        diffs, same = [], True
        for want, answer, question in zip(ref["answers"], mine["answers"], record["questions"]):
            labels = [option["label"] for option in question["options"]]
            order = [labels.index(option) for option in want["option_ids"]]
            diffs.append(max(abs(answer["probs"][i] - want["probs"][k]) for k, i in enumerate(order)))
            pick = order[max(range(len(order)), key=lambda k: want["probs"][k])]
            same_q = pick == max(range(len(answer["probs"])), key=lambda i: answer["probs"][i])
            same = same and same_q
            agree += same_q
            total += 1
        worst = max(worst, max(diffs))
        results.append({"name": name, "questions": len(questions), "tokens": mine["input_tokens"],
                        "input_ids_identical": same_ids, "max_abs_prob_diff": max(diffs),
                        "decisions_agree": same, "engine_ms": mine["engine_ms"],
                        "reference_s": ref.get("seconds")})
        print(f"{name:38s} q={len(questions):2d} tok={mine['input_tokens']:5d} ids={'same' if same_ids else 'DIFF'} "
              f"max|dp|={max(diffs):.1e} agree={'yes' if same else 'NO '} engine {mine['engine_ms'] / 1e3:6.1f} s",
              flush=True)
    print(f"\n{len(results)} requests, {total} questions: input ids identical on {ids_same}/{len(results)}, "
          f"decisions agree on {agree}/{total}; max |dp| {worst:.2e} (reference {reference['dtype']})")
    engine_s = [r["engine_ms"] / 1e3 for r in results]
    print(f"engine per request: median {statistics.median(engine_s):.1f} s, total {sum(engine_s):.0f} s "
          f"({wall:.0f} s with the load)")
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")
    return 0 if agree == total and ids_same == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
