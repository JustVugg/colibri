# Brio mode — scoring a closed set instead of generating

In brio mode the engine stops writing and starts **scoring**. You give it a prompt
and a set of allowed options; it answers with how likely each option is, and with
an entropy that says how sure it is. Same binary, same model, same chat path: the
mode is a key on the request, not a build flag or a separate server.

## Why you would want it

A generated answer is a string. You parse it, you hope it is one of the values you
asked for, and you get nothing back about confidence — the model writes `high` the
same way whether it knows or is guessing.

Brio mode answers the three questions a closed-set decision actually has:

| | generation | brio |
|---|---|---|
| which option | a string to parse | the option, by construction |
| how likely each one | not available | a probability per option, including ones the model would never write |
| does it know | not available | an entropy: 0 means one plausible option, 1 means all of them |

The third row is the one you cannot get any other way, and it is usually the one
that decides whether a decision can be automated.

## It needs a running server

There are three ways to use brio mode and they are all clients of the same
server: the **HTTP endpoint**, the **terminal** (`coli chat --attach`) and the
**Brio page** in the web interface. What does not exist is a one-shot form —
no `coli brio <model> ...` that loads, answers and exits.

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
| your own code | `POST /v1/brio`, below |
| the terminal | `coli chat --attach http://127.0.0.1:8000`, then `/brio` |
| the browser | open the server's address, Brio in the navigation dock |

All three end up in the same place, so a snapshot warmed by one of them is
already warm for the others.

## The HTTP endpoint

```
POST /v1/brio
```

```json
{
  "model": "qwen36",
  "state": "The pull request changes the logprob maths in the engine. 340 lines, 8 files, no tests. CI is green but the project has no coverage on that path.",
  "question": "What should the reviewer do?",
  "options": ["merge", "request changes", "close"]
}
```

| field | required | meaning |
|---|---|---|
| `model` | yes | as in every other endpoint |
| `options` | one of | 2 to 64 distinct non-empty strings: one closed question |
| `questions` | one of | an array of `{question, options}`: many questions on one state, see below |
| `schema` | one of | an object `field: [values]`: a JSON object filled one field at a time, see below |
| `task` | optional | with `schema`, what the object is for |
| `state` | one of | the text to decide on |
| `messages` | one of | a chat history used as the context instead of `state` |
| `question` | optional | what to ask about the state |
| `normalize` | optional | `mean` (default) or `sum`, see below |
| `cache_slot` | optional | forced KV slot; by default derived from `state` |

The reply:

```json
{
  "object": "brio.choice",
  "answer": "request changes",
  "entropy": 0.121,
  "normalize": "mean",
  "choices": [
    {"option": "request changes", "p": 0.974, "logprob": -0.252, "mean_logprob": -0.126, "tokens": 2},
    {"option": "merge",           "p": 0.023, "logprob": -4.007, "mean_logprob": -4.007, "tokens": 1},
    {"option": "close",           "p": 0.004, "logprob": -5.841, "mean_logprob": -5.841, "tokens": 1}
  ],
  "usage": {"prompt_tokens": 88, "completion_tokens": 0, "read_tokens": 4, "total_tokens": 92}
}
```

`completion_tokens` is always **0**: nothing is generated. `read_tokens` counts the
option tokens the engine read to score them.

### Many questions on one text: `questions`

The document is photographed once and every question pays only for its own
words. This is the case where brio mode saves the most (5.7x against the chat,
measured below), and the server keeps the order of the snapshots itself.

```json
{
  "model": "qwen36",
  "state": "340 lines, 8 files, no tests. CI is green but no coverage on that path.",
  "questions": [
    {"question": "What should the reviewer do?", "options": ["merge", "request changes", "close"]},
    {"question": "Does it need tests?",          "options": ["yes", "no"]},
    {"question": "How risky is it?",             "options": ["high", "medium", "low"], "normalize": "sum"}
  ]
}
```

The reply is `brio.answers`: an `answers` array in the same order, each entry
shaped like a single `brio.choice` (`question`, `answer`, `entropy`, `choices`),
and one `usage` for the whole request. Up to 64 questions, each with 2 to 64
options; `normalize` can be set per question or once for all of them.

### Fill a JSON object: `schema`

The braces, the quotes and the field names are data the server writes. For
each field, in the order you give them, the model only picks one of the values
you allow, with the fields already filled visible to it. The JSON cannot come
out malformed and no value can be outside your list, because nothing is
generated.

```json
{
  "model": "qwen36",
  "state": "340 lines, 8 files, no tests. CI is green but no coverage on that path.",
  "task": "Review this pull request.",
  "schema": {
    "decision":    ["merge", "request changes", "close"],
    "needs_tests": ["yes", "no"],
    "risk":        ["high", "medium", "low"],
    "area":        ["engine", "gateway", "docs"]
  }
}
```

The reply is `brio.schema`: `json` is the filled object, ready to use, and
`fields` carries, per field, the chosen `value`, its `p`, the `entropy` of that
cell and the full `choices`. The entropy per field is the point: the measured
run below was sure about `area` (0.17) and not about `needs_tests` and `risk`
(0.95 and 0.99), and said so, where the chat wrote `"risk": "high"` with the
same face. `task` is optional. Field names cannot contain quotes, backslashes
or newlines; values can, they are escaped.

### From your own code

No SDK is needed: it is one JSON request on the same server, with the same
API key header as the rest.

```python
import requests
r = requests.post("http://127.0.0.1:8000/v1/brio", json={
    "model": "qwen36",
    "state": open("ticket.txt").read(),
    "questions": [
        {"question": "Which queue?", "options": ["billing", "bugs", "sales"]},
        {"question": "Urgent?",      "options": ["yes", "no"]},
    ]})
for a in r.json()["answers"]:
    print(a["question"], "->", a["answer"], f"(entropy {a['entropy']:.2f})")
```

```js
const r = await fetch("http://127.0.0.1:8000/v1/brio", {
  method: "POST", headers: {"Content-Type": "application/json"},
  body: JSON.stringify({ model: "qwen36", state: ticket,
    schema: { queue: ["billing", "bugs", "sales"], urgent: ["yes", "no"] } })
});
const { json, fields } = await r.json();
// json.queue, json.urgent are guaranteed to be values from your lists
```

### Do not put the options in the prompt

Write the state and the question; leave the option list to the `options` field. On a
real case that list was 48 tokens of 123, and it is about half of what the mode
saves. Naming the options in the text also biases the scoring towards whichever one
the sentence happens to mention last.

### `mean` or `sum`

`logprob` is the log probability of the whole option string, so a two-token option
is penalised against a one-token option simply for being longer. Measured on qwen36:
summing picked `merge` where the same model's own greedy decoding said
`request changes`; the mean per token agreed with generation. `mean` is therefore
the default. `sum` is available when you want the literal probability of the string.

## Reading the entropy

Normalised to 0..1 over the number of options.

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

› /brio merge | request changes | close
  ✦ brio · 3 options · the model no longer generates, it assigns probabilities

› The PR touches the engine and carries no tests. What should we do?
  ◆ brio
     request changes  ███████████████████████░░░  93.6%  2 tok
     merge            ██░░░░░░░░░░░░░░░░░░░░░░░░   6.4%  1 tok
     close            ░░░░░░░░░░░░░░░░░░░░░░░░░░   0.0%  1 tok
     → request changes  entropy 0.218  (confident)
  58.71s · 4 tokens read · 0 generated

› /brio
  ✦ chat
```

`/brio` with options enters the mode with the conversation so far as the context;
`/brio` alone returns to chat. `:brio` works too. TAB completes the commands.

## In the browser

The **Brio** entry in the navigation dock opens a page built around the same
shape: the document on top, read once, and questions accumulating below it, each
with its own set of allowed options and its own answer. The bars show the
probability of every option, and the entropy sits next to the winner.

Options are per question, not shared across the page: "how risky is this" wants
low/medium/high where "do we sign" wants yes/no, and one list for all of them
would bend the questions to fit the list.

You can load the document from a file (plain text: `.txt`, `.md`, `.json`, `.csv`
and friends; PDF and Word are not supported and are refused rather than silently
read as noise). The page and the chat stay alive together: start a scoring run,
go and chat about something else, and the answers are waiting when you come back.

## Under the protocol

If you speak the [serve protocol](serve_protocol) directly, brio mode is two
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
discarded — on a one-token option, double the work.

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

That is exactly what `/v1/brio` does. Doing it in a client is possible and is how
the endpoint was prototyped, but three things are easy to get wrong: pinning, the
option list leaking into the prompt, and the length normalisation.

## Nested snapshots

The useful prefixes are nested: the shared instructions, and the instructions plus
this question. The engine keeps a few snapshots (`COLI_PIN_SLOTS`, default 4) and
always restores the deepest one that is a strict prefix of the new prompt.

With a single snapshot you have to choose which level to keep, and the other is paid
again on every request — measured on qwen36, 496 tokens per item instead of 176.

Send `pin=1` wherever you want a return point. The engine matches by token ids, so
nothing needs to be declared in advance, and it refuses a snapshot whose attention
rows the state no longer holds.

## Cost, measured

qwen36 (22 GB, 40 layers) over the gateway, one KV slot:

| task | brio | generation |
|---|---|---|
| one question, 3 options | 65.7 s, 0 tokens generated | 79.6 s, 5 generated |
| a 4-field JSON schema | 103.8 s, 104 tokens processed | 246.0 s, 226 processed |
| 4 items sharing one instruction block | 45.7 s per item | 149.2 s per item |

The gap widens with how much the alternative has to **write** and how much scaffolding
it has to **re-read**. On the JSON case, generation also invented two field names that
were not in the schema; brio mode cannot, because the field names are yours and only
the values come from the model.

Note that these are on a disk-streaming engine, where the prefill costs about
0.9 s/token. The ratios are the point, not the absolute numbers.

## Limits

- **Options are scored, not validated.** Two options that tokenise identically are
  indistinguishable.
- **The snapshot lives in the engine process.** Restart the server and the first
  request pays the full prefill again.
- **One KV slot means one conversation at a time.** With several slots, requests
  sharing a `state` are routed to the same slot so they share the snapshot; with one,
  interleaved clients evict each other.
- **`normalize` is a policy, not a fact.** Neither mean nor sum is right for every
  option set; if your options have very different lengths, look at both.
