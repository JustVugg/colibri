# System One mode: typed decisions with calibrated probabilities

System One mode answers closed questions with a probability per allowed
option. There is one API for it, `POST /v1/systemone`, with the request and
the reply of TypeSafe's Jev: a state, typed questions (`noul`, `choice`,
`score`), and for every question a probability per option. A language model
answers by **scoring** instead of writing; a decision model such as
[Laya](laya.md) answers natively. Same server and same model as the chat: the
mode is the endpoint, not a build flag or a separate process.

## Why you would want it

A generated answer is a string. You parse it, you hope it is one of the values you
asked for, and you get nothing back about confidence: the model writes `high` the
same way whether it knows or is guessing.

System One mode answers the three questions a closed-set decision actually has:

| | generation | System One |
|---|---|---|
| which option | a string to parse | the option, by construction |
| how likely each one | not available | a probability per option, including ones the model would never write |
| does it know | not available | a `confidence` per answer, 1 when one option holds the mass and 0 when it is flat (the terminal and the page also show the entropy) |

The third row is the one you cannot get any other way, and it is usually the one
that decides whether a decision can be automated.

## It needs a running server

There are three ways to use System One mode and they are all clients of the same
server: the **HTTP endpoint**, the **terminal** (`coli chat --attach`) and the
**System One page** in the web interface. What does not exist is a one-shot form:
no `coli decide <model> ...` that loads, answers and exits.

That is not an omission. The whole point is the snapshot: the shared prefix is
read once and kept in the engine's memory, so every question after the first is
cheap. A process that exited after one answer would throw away the thing the mode
exists for, and would be slower than plain generation for the trouble.

So start the server once and keep it:

```bash
cd c
COLI_MODEL=/nvme/qwen36 ./coli serve --host 127.0.0.1 --port 8000 --model-id qwen36
```

and then reach it however you prefer:

| from | how |
|---|---|
| your own code | `POST /v1/systemone`, below |
| the terminal | `coli chat --attach http://127.0.0.1:8000`, then `/decide` |
| the browser | open the server's address, System One in the navigation dock |

All three end up in the same place, so a snapshot warmed by one of them is
already warm for the others.

## The HTTP endpoint

```
POST /v1/systemone
```

| field | required | meaning |
|---|---|---|
| `state` | yes | what the questions are about: text, an object or an array (JSON is read as text) |
| `questions` | yes | an object of 1 to 64 questions, `id: question`; the reply uses the same ids |
| `model` | no | any name: the served model answers (a Jev client sends `jev-latest`) |
| `normalize`, `pin_state`, `prefix`, `cache_slot` | no | colibri's options, below |

Every question has a `type`, optional `instructions` (text, object or array)
and its `criteria`:

| type | criteria | the answer |
|---|---|---|
| `noul` | optional `{"true": ..., "false": ...}` descriptions | `noul`: the probability of yes |
| `choice` | `{label: description or null}`, 1 to 255 labels | `choice`, `probabilities` by label, `confidence` |
| `score` | `[level 0, level 1, ...]`, 1 to 255 levels | `score` (the expected level), `legend` and `probabilities` keyed `"0".."n-1"`, `confidence` |

```bash
curl -s http://127.0.0.1:8000/v1/systemone \
  -H "Authorization: Bearer $COLI_API_KEY" -H "Content-Type: application/json" \
  -d '{
    "state": "Hi, I have been trying to connect my Stripe account for 3 days and the integration keeps failing. I am losing sales. Please help ASAP.",
    "model": "jev-latest",
    "questions": {
      "urgency":    {"type": "noul",   "instructions": "Does this message express urgency?"},
      "department": {"type": "choice", "instructions": "Which team should handle this?",
                     "criteria": {"billing": "payments, invoices, Stripe payouts",
                                  "technical": "bugs, outages, integration errors",
                                  "sales": "pricing, plans, upgrades"}},
      "severity":   {"type": "score",  "instructions": "How severe is the customer impact?",
                     "criteria": ["no impact", "minor inconvenience", "blocked on one task",
                                  "losing money", "business down"]}
    }}'
```

The reply, from Qwen3.6-35B-A3B on a CPU box (1m46 with the experts streamed
from disk, the state read once for the three questions):

```json
{"id": "req_5f0c...", "model": "qwen36", "provider": "colibri",
 "answers": {
   "urgency":    {"type": "noul", "noul": 0.847606},
   "department": {"type": "choice", "choice": "technical",
                  "probabilities": {"billing": 0.059806, "technical": 0.939983, "sales": 0.000211},
                  "confidence": 0.909974},
   "severity":   {"type": "score", "score": 2.831772,
                  "legend": {"0": "no impact", "1": "minor inconvenience", "2": "blocked on one task",
                             "3": "losing money", "4": "business down"},
                  "probabilities": {"0": 0.044403, "1": 0.02624, "2": 0.034899, "3": 0.842097, "4": 0.052361},
                  "confidence": 0.802621}},
 "usage": {"input_tokens": 86, "output_tokens": 15, "cost": 0}}
```

`usage.output_tokens` counts the option tokens a language model **read**:
nothing is generated. The probabilities of a question always sum to 1 over
its options.

### Many questions on one text

The document is photographed once and every question pays only for its own
words. This is the case where System One mode saves the most (5.7x against the
chat, measured below), and the server keeps the order of the snapshots itself:
the state alone first, then each question on top of it.

A closed question with a list of answers is one `choice` whose labels are the
answers (`{"merge": null, "request changes": null, "close": null}`); a JSON
object whose fields each take one of a few values is one `choice` per field.

Up to 1.12.1 the same channel also had its own route, `POST /v1/brio`, with an
`options`, a `questions` and a `schema` form. It is gone: `/v1/systemone` is
the one decision API. A `schema` filled its fields in order, each seeing the
values chosen before it; questions on `/v1/systemone` are answered each on
its own, from the state.

### From your own code

No SDK is needed: it is one JSON request on the same server, with the same
API key header as the rest.

```python
import requests
r = requests.post("http://127.0.0.1:8000/v1/systemone", json={
    "state": open("ticket.txt").read(),
    "questions": {
        "queue":  {"type": "choice", "instructions": "Which queue?",
                   "criteria": {"billing": None, "bugs": None, "sales": None}},
        "urgent": {"type": "noul", "instructions": "Is it urgent?"},
    }})
answers = r.json()["answers"]
print(answers["queue"]["choice"], answers["queue"]["confidence"], answers["urgent"]["noul"])
```

```js
const r = await fetch("http://127.0.0.1:8000/v1/systemone", {
  method: "POST", headers: {"Content-Type": "application/json"},
  body: JSON.stringify({ state: ticket, questions: {
    queue: { type: "choice", instructions: "Which queue?", criteria: { billing: null, bugs: null, sales: null } },
    urgent: { type: "noul", instructions: "Is it urgent?" } } })
});
const { answers } = await r.json();
// answers.queue.choice is one of your labels; answers.urgent.noul is P(yes)
```

### Switching from Jev: change the base URL, keep your code

colibri answers `POST /v1/systemone` with the request and the reply of
TypeSafe's Jev API. Code written against Jev keeps working: point it at
colibri and change nothing else. The two official SDKs, unmodified, are
tested against `coli serve` on a language model and on a decision engine
(`tests/test_jev_sdk.py`, `make -C c jev-sdk-check`).

Python, `typesafe_sdk`:

```python
from typesafe_sdk import TypeSafeClient

client = TypeSafeClient(api_key="your-colibri-key", base_url="http://127.0.0.1:8000")
result = client.system_one(model="jev-latest", state=ticket, questions=questions)
print(result.choices["department"].choice, result.nouls["urgency"].noul)
```

TypeScript, `@typesafe-ai/sdk`:

```ts
import { TypeSafeClient } from "@typesafe-ai/sdk";

const client = new TypeSafeClient({ apiKey: "your-colibri-key", baseURL: "http://127.0.0.1:8000" });
const { answers } = await client.systemOne({ state: ticket, questions });
```

Or leave the code as it is and set `TYPESAFE_BASE_URL=http://127.0.0.1:8000`
and `TYPESAFE_API_KEY` (the server's `COLI_API_KEY`; the SDKs want a
non-empty key, and a server started without one accepts any). The SDKs
POST to `<base_url>/v1/systemone` and list models at `<base_url>/v1/models`;
both routes accept any `model` name (a Jev client sends `jev-latest`).

### What a language model is asked

How the three primitives map onto a language model, so you know what it is
actually asked:

| Jev primitive | what is scored | what goes into the question text | reply |
|---|---|---|---|
| `noul` | `yes` / `no` | `instructions`, then `yes: <criteria.true>` and `no: <criteria.false>` when given, then "Answer yes or no." | `noul` = probability of yes |
| `choice` | the labels of `criteria` (up to 255) | `instructions`, then one line per label with its description | `choice`, `probabilities` by label, `confidence` |
| `score` | the level numbers `"1".."n"` (1 to 255 levels) | `instructions`, then `k: <description>` per level, then "Answer with the number." | `score` = expected level under the distribution, `legend` and `probabilities` keyed `"0".."n-1"` as Jev numbers them, `confidence` |

`state`, `instructions` and every criterion may be a string, an object or an
array; JSON is serialized as text. `confidence` is the formula their
documentation gives, `(n * peak - 1) / (n - 1)`: 1 when all the mass sits on
one label, 0 when flat; `noul` carries none, as in theirs. A question with a
single label or level is answered without asking the model.

What differs from Jev, stated rather than hidden:

- `model` is the served model, `provider` is `"colibri"`, `usage.cost` is 0,
  and `id` is the request id (also the `x-typesafe-request-id` header).
- `usage.output_tokens` counts the option tokens a language model **read**;
  nothing is generated. `input_tokens` is the longest prompt of the request.
  A decision engine reports the tokens it read as `input_tokens` and 0 output.
- Validation errors are `422`, as theirs, with this server's error envelope
  (`error.message`, `error.param`) and Jev's `detail` list beside it.
- `GET /v1/models` lists the Jev card (`models`: name, description,
  release_date) next to OpenAI's `data`.
- The probabilities are this model's, normalised over the options with the
  `sum` rule below. The numbers will not match Jev's on the same input; the
  contract is the same, the model is yours.

### colibri's options

None of these is sent by a Jev client; each is optional.

| field | meaning |
|---|---|
| `normalize` | `"sum"` (default) or `"mean"`: how a language model's option log-probabilities become one score per option, see below |
| `pin_state` | `true` / `false`: photograph the state for the next request. Left out, the server photographs it when two or more questions read it, or when the same state came back from an earlier request. A state that changes on every call (a game, a sensor feed) is read straight through, one engine call fewer per request |
| `prefix` | a fixed text read before the state, photographed once per KV slot and retaken if the engine evicted it. For rules or instructions that stay while the state changes: on the tiny MiMo fixture, rules of 690 positions plus a frame cost 143 positions per request instead of 690 |
| `cache_slot` | the KV slot, as on the chat endpoints; by default it is derived from the state (from the prefix, when there is one) |

A decision engine has no log-probabilities and no photos: it ignores
`normalize` and `pin_state` and refuses `prefix` (put that text in `state` or
`instructions`); `cache_slot` reaches it as the DECIDE slot.

### Do not put the options in the state

Write the state and the instructions; leave the options to `criteria`. The
server lists a choice's labels once, in the question, and reads each one as
the answer's continuation. Repeating them in the state or the instructions
makes the prompt longer for every option read, and naming them in a sentence
biases the scoring towards whichever one the sentence happens to mention last.

### `sum` or `mean`

`logprob` is the log probability of the whole option string; `mean_logprob`
divides it by the option's tokens. `sum` is the default. Measured on a
30-case code-review benchmark (ALLOW / REVIEW / DENY), `mean` answered DENY on
all 30, a one-word README fix included, because DENY is two tokens and ALLOW
one: a per-token average favours whichever option has more tokens. With `sum`
the same suite scored ALLOW 10/10 and DENY 9/10. `mean` stays available, and
the server warns when it is asked for options whose token counts differ. On
qwen36 one case went the other way (`sum` picked `merge` where the model's
own greedy answer was `request changes`), so when your options have very
different lengths, look at both.

## Reading the answer

`confidence` is `(n * peak - 1) / (n - 1)` over the `n` options: 1 when all
the mass sits on one, 0 when it is flat. The terminal and the page show the
entropy of the same probabilities, normalised to 0..1 over the number of
options:

| entropy | reading |
|---|---|
| below 0.4 | the model is confident |
| 0.4 to 0.8 | unsure |
| above 0.8 | it does not know: treat the top option as a coin flip |

An entropy near 1 is a useful answer, not a failure. It is the model telling you
this decision needs a human, which a generated sentence never does.

## In the terminal

The conversation so far becomes the context, so you can chat, then switch to
scoring without restating anything.

```
coli chat --attach http://127.0.0.1:8000

› /decide merge | request changes | close
  ✦ System One · 3 options · the model no longer generates, it assigns probabilities

› The PR touches the engine and carries no tests. What should we do?
  ◆ System One
     request changes  ███████████████████████░░░  93.6%
     merge            ██░░░░░░░░░░░░░░░░░░░░░░░░   6.4%
     close            ░░░░░░░░░░░░░░░░░░░░░░░░░░   0.0%
     → request changes  entropy 0.218  (sure)
  58.71s · 131 prompt tokens · 4 option tokens read · nothing generated

› /decide
  ✦ chat
```

`/decide` with options enters the mode with the conversation so far as the context;
`/decide` alone returns to chat. `:decide` works too. TAB completes the commands.
Each question goes to `POST /v1/systemone` as one `choice` whose labels are the
options, with the conversation as the state.

## In the browser

The **System One** entry in the navigation dock opens a page built around the same
shape: the document on top, read once, and questions accumulating below it, each
with its own set of allowed options and its own answer. The bars show the
probability of every option, and the entropy sits next to the winner. Each
question goes to `POST /v1/systemone` as one `choice`, on a language model and
on a decision model alike.

Options are per question, not shared across the page: "how risky is this" wants
low/medium/high where "do we sign" wants yes/no, and one list for all of them
would bend the questions to fit the list.

You can load the document from a file (plain text: `.txt`, `.md`, `.json`, `.csv`
and friends; PDF and Word are not supported and are refused rather than silently
read as noise). The page and the chat stay alive together: start a scoring run,
go and chat about something else, and the answers are waiting when you come back.

## Under the protocol

If you speak the [serve protocol](serve_protocol) directly, System One mode is two
optional keys on `SUBMIT`. Both are opt-in: a request that does not send them
produces byte-identical frames to one sent before the feature existed.

```
SUBMIT <id> <slot> <bytes> <max_tokens> <temp> <top_p> [gbytes] [key=value ...]
```

| key | effect |
|---|---|
| `logprobs=k` | read the prefill out: one `ECHO` frame per fresh position, and a numeric tail on `DATA` |
| `pin=1` | photograph the engine state at the end of this prompt |

`max_tokens=0` is legal **only** together with `logprobs>0`, and means "read the
prompt and stop". Without it each option costs a full decode step that is then
discarded: on a one-token option, double the work.

An `ECHO` frame:

```
ECHO <id> <nbytes> <position> <logprob> <k> [<token_id> <logprob>]*k
<nbytes bytes of the token's text>
```

`<logprob>` is the log-softmax of the token that was actually at that position, over
the whole vocabulary. The first position of a fresh prompt has nothing to condition
on and carries `nan 0`; when a snapshot is restored, its saved logits supply that
first predictor instead.

### Scoring by hand

1. Send the shared prefix with `pin=1 logprobs=1 max_tokens=0`.
2. For each option, send `prefix + " " + option` with `logprobs=1 max_tokens=0`.
3. Sum the `ECHO` log probabilities at positions past the prefix, divide by the
   number of those positions, and take a softmax across the options.

That is what `/v1/systemone` does on a language model (with `sum` by
default instead of the per-token mean of step 3). Doing it in a client is
possible and is how the endpoint was prototyped, but three things are easy to
get wrong: pinning, the option list leaking into the prompt, and the
normalisation.

## Nested snapshots

The useful prefixes are nested: the shared instructions, and the instructions plus
this question. The engine keeps a few snapshots (`COLI_PIN_SLOTS`, default 4) and
always restores the deepest one that is a strict prefix of the new prompt.

With a single snapshot you have to choose which level to keep, and the other is paid
again on every request: measured on qwen36, 496 tokens per item instead of 176.

Send `pin=1` wherever you want a return point. The engine matches by token ids, so
nothing needs to be declared in advance, and it refuses a snapshot whose attention
rows the state no longer holds.

## What a snapshot holds

A snapshot is not always just token ids. It must hold everything a layer carries from one
token to the next, or the option that resumes from it starts from a state that never
existed, and the numbers stay plausible. What that is depends on the model:

- **Attention rows indexed by position** (every full-attention layer) stay where they
  are. Nothing written after the snapshot lands below it, so the snapshot holds only
  the ids, and the engine checks that the state still holds those ids before it trusts
  the rows. An unrelated prompt in between rewrites them; the snapshot is then
  released and the option is computed from the start.
- **Recurrent state** (the DeltaNet layers of Qwen3.6 and Qwen3.8) is copied into the
  snapshot, because a recurrence cannot be rewound.
- **Sliding-window rings** (MiMo-V2.6: 39 of its 48 layers attend the last 128
  positions) are copied too. A windowed layer keeps position p in slot p mod 128, so
  the first option writes its own tokens over the slots of the snapshot's last
  positions. Without the copy, the second option would read the first one's keys.
  On MiMo-V2.6 Flash that is about 50 MB per snapshot.

A prompt with a **picture** never takes a snapshot and never resumes from one. Its ids
describe where the picture goes, not what it shows.

**The read-out starts where the snapshot ends**, not where the shared text ends. Two
options share the text before them, and stopping there would leave the first option
token without the logits that predict it. On MiMo and GLM a prompt read out with no
snapshot to resume from is read out from position 0, never from a live prefix. That
is what `echo` on `/v1/completions` needs.

The serve tests hold an engine to both halves on its tiny fixture
(`tests/test_brio_serve.py`, and `tests/test_mimo_prefix_serve.py` for MiMo's rings and
pictures). An option read through the snapshot must give the numbers a cold engine gives,
to the last printed digit. It must also **process only its own tokens**: checking the
numbers alone would pass on an engine that quietly recomputed everything.

## Cost, measured

qwen36 (22 GB, 40 layers) over the gateway, one KV slot:

| task | System One | generation |
|---|---|---|
| one question, 3 options | 65.7 s, 0 tokens generated | 79.6 s, 5 generated |
| 4 items sharing one instruction block | 45.7 s per item | 149.2 s per item |

The gap widens with how much the alternative has to **write** and how much
scaffolding it has to **re-read**. (Measured on the route that preceded
`/v1/systemone`, with the same scoring channel and the same photographs.)

Note that these are on a disk-streaming engine, where the prefill costs about
0.9 s/token. The ratios are the point, not the absolute numbers.

## Limits

- **Options are scored, not validated.** Two options that tokenise identically are
  indistinguishable.
- **The snapshot lives in the engine process.** Restart the server and the first
  request pays the full prefill again.
- **Snapshots cost memory where they carry state.** `COLI_PIN_SLOTS` (default 4) bounds
  how many the engine keeps: about 50 MB each on MiMo-V2.6 Flash, tens of MB on Qwen3.6,
  only the ids and the final logits on engines with nothing else to keep.
- **One KV slot means one conversation at a time.** With several slots, requests
  sharing a `state` are routed to the same slot so they share the snapshot; with one,
  interleaved clients evict each other.
- **`normalize` is a policy, not a fact.** Neither mean nor sum is right for every
  option set; if your options have very different lengths, look at both.

## Decision engines

A language model answers `/v1/systemone` by scoring each option through its
logprob channel: a prompt per question, a read-out per option. A **decision
engine** is a model built for the question itself. It reads the state and the
typed questions and returns a probability per option in one forward pass,
with the calibration its authors fitted. colibri serves it on the same
endpoint and with the same reply, so a Jev client, the SDKs and the dashboard
cannot tell which kind answered.

| engine | model | doc |
|---|---|---|
| `c/laya` | Laya (Convai Innovations): ModernBERT encoder + decision head | [laya.md](laya.md) |
| `c/gliner_decide` | GLiNER2.5-Decide (fastino): DeBERTa-v3 encoder + GLiNER2's classification head | [gliner_decide.md](gliner_decide.md) |

A decision engine serves `POST /v1/systemone` only. Chat, completions and
messages answer 400 with a pointer to `/v1/systemone`, and
`/v1/models` lists the model with `capabilities: ["systemone", "decision"]`
(a language model says `["chat", "systemone"]`). The path is one round trip:
the gateway checks the request, sends it to the engine as one record, and
shapes the answer. Nothing is rendered into a prompt and no option is scored
on its own. On Laya the gateway's share is about 2 ms per request: 8.1 ms at
the client for three questions on the tiny fixture, 6.1 of them in the
engine; 882 ms on the real checkpoint, 879.5 in the engine. The engine's own
time is in the `x-colibri-engine-ms` header.

### The contract, for an engine author

Everything below is generic; `c/decide_serve.h` implements the engine side,
so a new engine writes only its model and one function.

**Handshake.** The engine announces itself on the `CAPS` line of the serve
protocol: `decide=1` (it takes `DECIDE`) and `chat=0` (it generates nothing).
An engine that also chats leaves `chat` out; the gateway then sends
`/v1/systemone` to `DECIDE` and the chat endpoints to `SUBMIT` as before.

**Request.** `DECIDE <id> <slot> <bytes>` followed by one JSON record:

```json
{"state": "{\"subject\": \"Duplicate charge\", \"body\": \"...\"}",
 "state_type": "object",
 "questions": [
   {"id": "department", "type": "choice", "instructions": "Which department?",
    "options": [{"label": "billing", "text": "invoices, payments, refunds"},
                {"label": "other", "text": null}]},
   {"id": "urgency", "type": "score", "instructions": "How urgent is this?",
    "options": [{"label": "0", "text": "not urgent"}, {"label": "1", "text": "blocking"}]},
   {"id": "churn", "type": "noul", "instructions": "Does the user threaten to leave?",
    "options": [{"label": "false", "text": null}, {"label": "true", "text": null}]}]}
```

- Every text is already text. A string is kept exactly as sent; an object or
  an array (a state, instructions, a criterion) is written as
  `json.dumps(value, ensure_ascii=False)`, the serialization the reference
  packages apply, so the engine reads the bytes its model was trained on.
  `state_type` says what the caller sent (`string`, `object`, `array`, ...),
  because a model may treat a list (a conversation) differently.
- Options come in the caller's order: a choice's labels with their
  descriptions (`null` for none), a score's levels from 0, a noul's `false`
  then `true` with the optional criteria. Missing instructions get the same
  default text the language-model path asks with.
- `slot` is the KV slot, for an engine that keeps state between requests.

**Answer.** One `DECISION <id> <bytes>` frame, then `DONE`:

```json
{"answers": [{"id": "department", "logits": [4.68, -2.78], "probs": [0.9804, 0.0196],
              "temperature": 1.906, "actions": {"act": 1.0, "escalate": 0.0},
              "tokens": 96, "state_tokens": 50, "state_dropped": 0}],
 "input_tokens": 348, "engine_ms": 879.5}
```

`probs` are the engine's final probabilities, one per option in the record's
order; the gateway builds the Jev answer from them alone (argmax for
`choice`, the expected level for `score`, `probs[1]` for `noul`). `logits`,
`temperature` and `actions` (a policy head such as Laya's act / escalate) are
the engine's own record of how it got there. A record the engine refuses is
`ERROR <id> DECIDE_INVALID <field>: <reason>`, which the client receives as a
422 naming that field; `DECIDE_FAILED` is the engine's own failure (500).

**Registry.** A family with `modality="decision"` and
`FamilyCapabilities(decision=True)` in `c/family_registry.py`, and a way for
`resolve_model` to recognise its checkpoint (Laya: `rl_agent_config.json`
plus `encoder/config.json`, keyed `laya_<encoder model_type>`;
GLiNER2.5-Decide: a `config.json` whose `model_type` is `extractor` plus
`encoder_config/config.json`, keyed
`gliner2_<architecture>_<encoder model_type>`). `checkpoint_files` names the
files `coli doctor` checks. `coli serve`
and `coli web` then serve it, `coli info` and `coli plan` describe it, and
`coli chat` / `coli run` refuse it with a pointer to the endpoint.

**Tests.** A tiny fixture whose reference answers come from the model's own
package (`tools/make_laya_tiny.py`, `tools/make_laya_ref.py`,
`tests/test_laya_tiny.py`; `tools/make_gliner_decide_tiny.py`,
`tools/make_gliner_decide_ref.py`, `tests/test_gliner_decide_tiny.py`), and
the fixture behind the real gateway (`tests/test_decision_serve.py`,
`tests/test_jev_sdk.py`).

### The next one

- **Clef** (a joint schema head over Qwen3.8-27B, dense, with vision): the
  backbone is the qwen36 engine's. It would answer `DECIDE` in that engine's
  serve loop next to `SUBMIT`, announce `decide=1` without `chat=0` (the same
  process still chats), run the backbone over the rendered record and apply
  the head to its final hidden states. The record already carries what a
  schema head needs (every question with its typed options); a picture in the
  state would need the record to carry image parts, which today it does not.
