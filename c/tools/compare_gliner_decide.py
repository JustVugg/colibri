#!/usr/bin/env python3
"""gliner_decide.c against the `gliner2` package on the real checkpoint (maintainer tool).

Runs the same requests through both and reports, per request and overall,
whether the token ids are the same, the largest probability difference,
whether every decision (argmax) agrees, and the latency of each on this
machine. The requests are every example of the model card
(fastino/GLiNER2.5-Decide README), written as System One questions, and cases
chosen to reach every path: several questions in one sequence, a document of
about 960 tokens, a conversation, JSON states and instructions, 2 to 40
options, label descriptions, unicode, emoji, URLs and e-mails, and the
GLiNER2 marker tokens inside the text.

Each request goes through the gateway's record builder
(openai_server.systemone_decision_record), as POST /v1/systemone sends it, and
reaches the package as the classify_text() call tools/make_gliner_decide_ref.py
builds (decision_tasks). Needs torch, transformers < 5 and gliner2 2.0.0; the
checkpoint is a local directory.

    python tools/compare_gliner_decide.py --model ~/models/GLiNER2.5-Decide \\
        --engine ./gliner_decide --threads 6
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

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "tools"))
from openai_server import systemone_decision_record  # noqa: E402
import make_gliner_decide_ref as reference  # noqa: E402

ENGINE_MAX_LEN = 4096           # the engine's default COLI_GLINER_MAX_LEN


def choice(labels, instructions=None):
    """A choice question: labels as a list (no descriptions) or a dict."""
    criteria = labels if isinstance(labels, dict) else {label: None for label in labels}
    question = {"type": "choice", "criteria": criteria}
    if instructions:
        question["instructions"] = instructions
    return question


def score(levels, instructions=None):
    question = {"type": "score", "criteria": levels}
    if instructions:
        question["instructions"] = instructions
    return question


def noul(instructions, true=None, false=None):
    question = {"type": "noul", "instructions": instructions}
    criteria = {key: value for key, value in (("true", true), ("false", false)) if value}
    if criteria:
        question["criteria"] = criteria
    return question


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
    "agreement is governed by the laws of the State of New York. ") * 4

BANKING = ["activate_my_card", "age_limit", "apple_pay", "atm_support", "automatic_top_up",
           "balance_not_updated", "bank_transfer_charge", "beneficiary_not_allowed",
           "cancel_transfer", "card_about_to_expire", "card_acceptance", "card_arrival",
           "card_delivery_estimate", "card_linking", "card_not_working", "card_payment_fee",
           "card_payment_not_recognised", "card_payment_wrong_rate", "card_swallowed",
           "cash_withdrawal_charge", "change_pin", "compromised_card", "contactless_not_working",
           "country_support", "declined_card_payment", "declined_cash_withdrawal",
           "declined_transfer", "direct_debit_not_recognised", "disposable_card_limits",
           "edit_personal_details", "exchange_charge", "exchange_rate", "exchange_via_app",
           "extra_charge_on_statement", "failed_transfer", "fiat_currency_support",
           "get_disposable_virtual_card", "get_physical_card", "getting_spare_card",
           "lost_or_stolen_card"]


def model_card():
    """Every classify_text() example of the model card, as System One questions.
    A multi-label task becomes one noul per label (System One has no
    multi-label question); the rest keep the card's labels and order."""
    review = ("Battery dies before lunch, but the keyboard and the screen are the best I have "
              "used on a laptop.")
    return [
        ("card: customer support intent",
         "My subscription renewed on April 15 for ¥5,400 after the service was already down. Can I "
         "get that charge refunded?",
         {"intent": choice(["order_status", "refund_request", "cancel_subscription", "update_payment",
                            "login_problem", "shipping_delay", "bug_report", "speak_to_human", "other"])}),
        ("card: banking request",
         "The transfer I sent this morning is still pending, and I think I used the wrong sort code. "
         "Can you stop it and add Emily as the beneficiary instead?",
         {"intent": choice(["transfer_pending", "transfer_cancel", "beneficiary_add", "card_lost",
                            "balance_inquiry", "fraud_report", "mortgage_application",
                            "fee_explanation"])}),
        ("card: travel request",
         "I need to move my Friday flight to Paris to Saturday morning, same cabin, and keep the "
         "aisle seat if you can.",
         {"request": choice(["book", "change", "cancel", "status", "seat_change", "refund", "baggage"])}),
        ("card: clinic request",
         "The rash came back after the antibiotics finished. Can I get a same-week appointment with "
         "dermatology, or should I just refill the cream?",
         {"request": choice(["book_appointment", "refill_prescription", "test_results", "referral",
                             "billing_question", "cancel_appointment"])}),
        ("card: review sentiment", review,
         {"sentiment": choice(["positive", "negative", "mixed", "neutral"])}),
        ("card: product aspects (one noul each)", review,
         {f"aspect_{name}": noul(f"Does the review talk about the {name}?")
          for name in ("battery", "keyboard", "screen", "camera", "price", "support")}),
        ("card: news topic",
         "The central bank held rates and said inflation is still above target, pushing bank stocks "
         "lower in afternoon trading.",
         {"topic": choice(["politics", "business", "sports", "science", "entertainment", "world"])}),
        ("card: document type",
         "INVOICE 1842\nBill to: Northstar QA\nAmount due: 2,400 USD\nDue: 30 April 2026\nWire "
         "instructions are on page 2.",
         {"document_type": choice(["invoice", "receipt", "contract", "resume", "support_email",
                                   "meeting_notes"])}),
        ("card: email triage, three heads",
         "From: compliance@group.example\nSubject: Protocol update — action required today\n\n"
         "Please confirm the new retention rule is applied before Friday's audit.",
         {"intent": choice(["fyi", "request", "approval", "complaint", "newsletter", "security_alert"]),
          "urgency": choice(["low", "normal", "high", "critical"]),
          "route": choice(["support", "billing", "legal", "security", "finance", "archive"])}),
        ("card: ticket routing",
         "[subject] 401k deduction missing from this paystub\n[body] Last month's contribution "
         "posted. This month the line is gone and HR told me to open a ticket.",
         {"queue": choice(["payroll", "benefits", "it_access", "facilities", "expense_reimbursement",
                           "manager_approval"])}),
        ("card: handoff to a person",
         "This is the third time I have explained the same missing refund. Stop the bot and get me "
         "a person.",
         {"handoff": choice(["yes", "no"]),
          "handoff_noul": noul("Should the conversation be handed to a person?")}),
        ("card: did the agent finish",
         "Goal: email the Q4 summary to every partner.\nLast action: draft saved in the hub.\nSend "
         "button is still disabled because two partners have no address.",
         {"finished": choice(["yes", "no"]), "finished_noul": noul("Did the agent finish the goal?")}),
        ("card: moderation",
         "Post the customer's home address in the public thread so everyone can see where the "
         "package actually went.",
         {"policy": choice(["allow", "personal_data", "harassment", "scam", "violence", "spam"])}),
        ("card: incident severity",
         "The deploy left resource tags inconsistent across staging. Production checkout is "
         "unaffected. No customer reports yet.",
         {"severity": choice(["info", "low", "medium", "high", "critical"]),
          "severity_score": score(["info", "low", "medium", "high", "critical"],
                                  "How severe is the incident?")}),
        ("card: urgency score 0-5",
         "Payroll file has to be corrected before the 5pm cutoff or the whole company is paid late.",
         {"urgency": score([None] * 6)}),
        ("card: spam or not",
         "Your mailbox is almost full. Click here in the next hour or we will delete every message.",
         {"label": choice(["spam", "ham"])}),
        ("card: several decisions at once",
         "Guest in room 1408 says the AC has been out since yesterday and they want to move tonight "
         "or leave. They also asked for the incidentals hold to be released.",
         {"intent": choice(["maintenance", "room_change", "checkout", "billing", "complaint",
                            "amenity_request"]),
          "priority": choice(["low", "normal", "high", "urgent"]),
          "needs_human": noul("Does this need a human?"),
          **{f"topic_{name}": noul(f"Is {name} one of the topics?")
             for name in ("hvac", "billing", "housekeeping", "noise", "safety")}}),
        ("card: question over a passage",
         "The treaty was signed in Paris in 1992. It entered into force the following year, after "
         "the last signatory ratified it.",
         {"answer": noul("Did the treaty enter into force in 1992?"),
          "answer_choice": choice(["yes", "no"], "Did the treaty enter into force in 1992?")}),
        ("card: book",
         "She closed the ledger, blew out the lamp, and listened for the stair. The house had been "
         "empty since the winter the river took the bridge.",
         {"genre": choice(["mystery", "romance", "history", "science_fiction", "literary_fiction",
                           "cookbook"])}),
        ("card: labels with a description",
         "Please reset the card PIN. The new one never arrived and the old one is locked after three "
         "tries.",
         {"intent": choice({"card_pin_change": "The customer wants a new PIN or the current PIN replaced",
                            "card_lost": "The physical card is missing",
                            "balance_inquiry": "The customer wants the current balance"})}),
        ("card: ordinal score 0-10",
         "I finished it in two nights. The ending is earned, the middle drags, and I would still "
         "hand it to a friend.",
         {"rating": score([None] * 11)}),
    ]


def own_cases():
    billed = ("Hi, we were billed twice for March. Please refund the duplicate today or we will "
              "cancel our plan.")
    department = choice({"billing": "invoices, payments, refunds",
                         "technical": "bugs, outages, system errors", "other": "everything else"},
                        "Which department should handle this?")
    return [
        ("jev quickstart, three types", billed,
         {"department": department,
          "urgency": score(["not urgent", "soon", "blocking"], "How urgent is this?"),
          "churn_risk": noul("Does the user threaten to cancel or leave?")}),
        ("jev object state", {"from": "user@acme.com", "subject": "Duplicate charge on invoice #4411",
                              "body": billed},
         {"department": department,
          "refund": noul("Does the user explicitly request a refund?",
                         true="the customer asks for money back", false="no refund is mentioned")}),
        ("stripe outage, sdk example",
         "Hi, I have been trying to connect my Stripe account for 3 days and the integration keeps "
         "failing. I am losing sales. Please help ASAP.",
         {"urgency": noul("Does this message express urgency?"),
          "department": choice({"billing": "payments, invoices, Stripe payouts",
                                "technical": "bugs, outages, integration errors",
                                "sales": "pricing, plans, upgrades"}, "Which team should handle this?"),
          "severity": score(["no impact", "minor inconvenience", "blocked on one task", "losing money",
                             "business down"], "How severe is the customer impact?")}),
        ("long contract, about 960 tokens", {"document": CONTRACT},
         {"auto_renewal": noul("Does the contract renew automatically?"),
          "law": choice({"new_york": "New York", "california": "California", "england": "England"},
                        "Which law governs the contract?"),
          "risk": score(["low", "medium", "high"], "How risky is this contract for the customer?")}),
        ("conversation",
         [{"role": "user", "content": "Hi, I cannot log in to my account since yesterday."},
          {"role": "assistant", "content": "Sorry about that. Did you try to reset the password?"},
          {"role": "user", "content": "Yes, twice, and the reset email never arrives."},
          {"role": "user", "content": "If it is not fixed today I will cancel the subscription."}],
         {"churn": noul("Does the user threaten to cancel?"),
          "topic": choice({"login": "access to the account", "billing": "payments",
                           "other": "anything else"}, "What is the conversation about?")}),
        ("json order", {"order": {"id": 98123, "total": 249.99, "currency": "EUR", "paid": True,
                                  "items": [{"sku": "TV-55", "qty": 1}, {"sku": "HDMI-2", "qty": 2}],
                                  "complaint": "the TV arrived with a cracked screen"}},
         {"refund_eligible": noul("Should this order be refunded?"),
          "category": choice({"damaged": "the product arrived broken", "late": "delivery delay",
                              "wrong_item": "a different product arrived", "none": "no problem"},
                             "What kind of problem is this?")}),
        ("json instructions",
         "We are moving our 300 seats to the annual plan if you can match the competitor's price.",
         {"opportunity": choice({"upsell": "more seats or a bigger plan", "renewal": "same plan again",
                                 "churn": "leaving"},
                                {"task": "classify the sales opportunity", "seats": "consider size"})}),
        ("forty banking intents", "My card payment was declined at the supermarket this morning.",
         {"intent": choice(BANKING, "Which banking intent is this?")}),
        ("twelve emotions", "I can't believe they cancelled the show, I'm so upset right now.",
         {"emotion": choice(["sadness", "joy", "love", "anger", "fear", "surprise", "disgust", "trust",
                             "anticipation", "neutral", "confusion", "gratitude"])}),
        ("ten-level score with descriptions",
         "Decent hotel, clean rooms, noisy street, breakfast could be better.",
         {"rating": score([f"{i} out of 10" for i in range(1, 11)], "Rate the stay from the review.")}),
        ("ten questions on one ticket",
         {"message": "Our whole team has been locked out since the SSO migration. Payroll runs "
                     "tomorrow and we need access today. This is the third outage this month."},
         {"department": department,
          "urgency": score(["not urgent", "soon", "blocking"], "How urgent is this?"),
          "churn": noul("Does the user threaten to leave?"),
          "sso": noul("Is single sign-on mentioned?"),
          "who": choice(["one_user", "a_team", "everyone"], "Who is affected?"),
          "repeat": noul("Has this happened before?"),
          "deadline": noul("Is there a deadline?"),
          "severity": score(["none", "minor", "major", "critical"], "How severe is the impact?"),
          "sentiment": choice(["calm", "worried", "angry"], "What is the tone?"),
          "channel": choice(["email", "phone", "chat"], "Which channel should answer?")}),
        ("italian", "Buongiorno, la fattura di settembre è sbagliata: ci avete addebitato due volte.",
         {"language": choice(["italian", "english", "german"], "Which language is the message in?"),
          "billing": noul("Is this about billing?")}),
        ("unicode mix",
         "Grüße aus München! ΟΔΟΣ ΣΟΦΙΑΣ, İstanbul, 我们的账户被锁了, café vs café, "
         "Ångström \U0001F642\U0001F44D\U0001F3FD",
         {"script": choice(["latin", "greek", "chinese", "mixed"], "Which script dominates?"),
          "friendly": noul("Is the message friendly?")}),
        ("emoji", "Best support ever \U0001F60D\U0001F64C fixed in 5 minutes!!!",
         {"sentiment": score(["angry", "neutral", "happy"], "How happy is the customer?")}),
        ("urls, e-mails, handles",
         "Please reset my password at https://accounts.example.com/reset?u=42 or mail "
         "Help.Desk@Example.COM; @support_team never answered on www.example.org/forum.",
         {"channel": choice(["website", "email", "social"], "Where did the user try to get help?"),
          "password": noul("Is this about a password?")}),
        ("marker tokens and whitespace",
         "Order [P] arrived  broken,   see\tphoto.\r\n\r\nThe [SEP_TEXT] courier said [L] it was "
         "fine [UNK] and [DESCRIPTION] we want a refund",
         {"damaged": noul("Was the item [L] damaged?"),
          "channel": choice({"email": "write an email [P] back", "phone": "call\tthe customer",
                             "none": None}, "Which channel  should   answer?")}),
        ("nli", {"premise": "A man is playing a guitar on stage in front of a crowd.",
                 "hypothesis": "A musician is performing."},
         {"relation": choice({"entailment": "the hypothesis follows", "neutral": "cannot tell",
                              "contradiction": "the hypothesis is false"},
                             "Does the premise entail the hypothesis?")}),
        ("empty state", "", {"anything": noul("Is there any request?")}),
    ]


def median(values):
    return statistics.median(values) if values else float("nan")


def package_answers(args, cases, records):
    """What the gliner2 package answers, unrounded, and how long classify_text takes."""
    import torch
    if args.threads:
        torch.set_num_threads(args.threads)
    model, version = reference.load_reference(args.model, allow_version=args.allow_version)
    recorder = reference.Recorder(model)
    out = []
    for (name, _, _), record in zip(cases, records):
        got = reference.run_case(model, recorder, record, ENGINE_MAX_LEN, min_gap=None)
        times = []
        if not args.skip_package_timing:
            tasks = reference.decision_tasks(record)
            kwargs = {} if got["words_used"] == got["words"] else {"max_len": got["words_used"]}
            model.classify_text(record["state"], tasks, **kwargs)            # warm
            for _ in range(args.repeat):
                started = time.perf_counter()
                model.classify_text(record["state"], tasks, **kwargs)
                times.append((time.perf_counter() - started) * 1e3)
        out.append({"name": name, "ids": got["ids"], "probs": [a["probs"] for a in got["answers"]],
                    "ms": times, "gliner2": version})
        print(f"  reference: {name}", file=sys.stderr, flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--engine", default=str(HERE / "gliner_decide"))
    parser.add_argument("--threads", type=int, default=0, help="both sides; 0 = their defaults")
    parser.add_argument("--repeat", type=int, default=3, help="timed runs per request")
    parser.add_argument("--json", help="write the per-request results here")
    parser.add_argument("--reference", help="the package's answers and timings: read from this "
                        "file when it exists, else computed and written to it")
    parser.add_argument("--skip-package-timing", action="store_true",
                        help="with a fresh --reference, compute the answers but do not time them")
    parser.add_argument("--allow-version", action="store_true",
                        help=f"run with an installed gliner2 other than {reference.PINNED_GLINER2}")
    args = parser.parse_args()

    cases = model_card() + own_cases()
    records = [systemone_decision_record({"state": s, "questions": q}) for _, s, q in cases]
    payloads = [{"payload": json.dumps(r, ensure_ascii=False)} for r in records]
    env = dict(os.environ, COLI_GLINER_MAX_LEN=str(ENGINE_MAX_LEN))
    if args.threads:
        env["OMP_NUM_THREADS"] = str(args.threads)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "records.json"
        path.write_text(json.dumps(payloads, ensure_ascii=False), encoding="utf-8")
        run = subprocess.run([args.engine, "--model", args.model, "--records", str(path), "--ids"],
                             capture_output=True, text=True, encoding="utf-8", env=env)
        if run.returncode:
            sys.exit(run.stderr)
        engine_out = [json.loads(line) for line in run.stdout.splitlines()]
        # every request repeat more times in one process, for the latency
        path.write_text(json.dumps(payloads * args.repeat, ensure_ascii=False), encoding="utf-8")
        run = subprocess.run([args.engine, "--model", args.model, "--records", str(path)],
                             capture_output=True, text=True, encoding="utf-8", env=env)
        if run.returncode:
            sys.exit(run.stderr)
    lines = [json.loads(line) for line in run.stdout.splitlines()]
    engine_ms = [[lines[r * len(cases) + i]["engine_ms"] for r in range(args.repeat)]
                 for i in range(len(cases))]

    if args.reference and Path(args.reference).exists():
        ref = json.loads(Path(args.reference).read_text(encoding="utf-8"))
    else:
        ref = package_answers(args, cases, records)
        if args.reference:
            Path(args.reference).write_text(json.dumps(ref) + "\n", encoding="utf-8")

    results, worst, agree, total, same_ids = [], 0.0, 0, 0, 0
    for (name, _, questions), got, want, times in zip(cases, engine_out, ref, engine_ms):
        if "error" in got:
            sys.exit(f"{name}: the engine refused: {got['error']}")
        ids_equal = got["sequence"]["ids"] == want["ids"]
        same_ids += ids_equal
        diffs, same = [], True
        for answer, p in zip(got["answers"], want["probs"]):
            diffs.append(max(abs(a - b) for a, b in zip(answer["probs"], p)))
            pick = max(range(len(p)), key=lambda i: (answer["probs"][i], -i))
            ref_pick = max(range(len(p)), key=lambda i: (p[i], -i))
            same = same and pick == ref_pick
            agree += pick == ref_pick
            total += 1
        worst = max(worst, max(diffs))
        results.append({"name": name, "questions": len(questions), "tokens": got["input_tokens"],
                        "ids_equal": ids_equal, "max_abs_prob_diff": max(diffs),
                        "decisions_agree": same, "engine_ms": median(times),
                        "package_ms": median(want["ms"])})
        print(f"{name:42s} q={len(questions):2d} tok={got['input_tokens']:5d} "
              f"ids={'same' if ids_equal else 'DIFF'} max|dp|={max(diffs):.1e} "
              f"agree={'yes' if same else 'NO '} engine {median(times):7.1f} ms   "
              f"package {median(want['ms']):7.1f} ms", flush=True)
    e = [r["engine_ms"] for r in results]
    pk = [r["package_ms"] for r in results]
    print(f"\n{len(results)} requests, {total} questions: token ids identical on {same_ids}/"
          f"{len(results)}, decisions agree on {agree}/{total}; max |dp| {worst:.2e}")
    print(f"latency per request (median over requests): gliner_decide {median(e):.1f} ms, "
          f"package on CPU {median(pk):.1f} ms; total {sum(e):.0f} vs {sum(pk):.0f} ms")
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")
    return 0 if agree == total and same_ids == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
