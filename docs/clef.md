# Clef on colibri

[Clef](https://huggingface.co/Cloudflare/clef), by Cloudflare (Apache-2.0), is
a decision model: give it a state (text or JSON) and typed questions
(`choice`, `score`, `noul`) and it returns a probability for every option of
every question in one forward pass. It is Qwen3.8-27B, post-trained, plus a
small transformer, the joint schema head, that reads the backbone's last
hidden states. Its model card calls its API compatible with Jev and System
One.

colibri runs it on the qwen36 engine, the one that already runs Qwen3.8-27B:
`POST /v1/systemone` is answered natively by the head, with the request and
the reply of TypeSafe's Jev API ([systemone.md](systemone.md)), and the same
process still chats.

## Download, convert, serve

```sh
hf download Cloudflare/clef --local-dir ~/Models/clef          # 55 GB, bf16
python3 c/tools/convert_qwen36.py --model ~/Models/clef --out ~/Models/clef_c
make -C c qwen36
./c/coli serve --model ~/Models/clef_c
```

The converter writes the backbone as for Qwen3.8-27B (an f16 container,
52 GB) and copies `joint_head_config.json` and `joint_head.safetensors` beside
it. `coli` recognises a qwen36 container that carries both files as Clef:
the banner says `Clef · 27B`, the API model id is `clef`, `coli info` adds a
`decision` line, `coli doctor` checks the head against the backbone, and
`/v1/models` lists `capabilities: ["chat", "systemone"]`.

Clef's backbone is not the stock one: compared tensor by tensor with
Qwen/Qwen3.8-27B, layers 0 to 39, the embeddings, the LM head and the vision
tower are identical, and the 186 projection and MLP matrices of layers 40 to
63 differ. A Qwen3.8-27B container cannot stand in for it.

```sh
curl -s localhost:8000/v1/systemone -H 'Content-Type: application/json' -d '{
  "model": "clef",
  "state": "Our checkout started returning errors and orders are blocked.",
  "questions": {
    "department": {"type": "choice", "instructions": "Which team should handle the message?",
                   "criteria": {"billing": "Payments or invoices", "technical": "Bugs or outages"}},
    "urgency": {"type": "score", "criteria": ["Can wait", "This week", "Today"]},
    "outage": {"type": "noul", "instructions": "Is a service down?"}}}'
```

```json
{"id": "req_973f...", "model": "clef", "provider": "colibri",
 "answers": {
   "department": {"type": "choice", "choice": "technical",
                  "probabilities": {"billing": 0.075526, "technical": 0.924474}, "confidence": 0.848948},
   "urgency": {"type": "score", "score": 1.837707, "legend": {"0": "Can wait", "1": "This week", "2": "Today"},
               "probabilities": {"0": 0.038915, "1": 0.084464, "2": 0.876622}, "confidence": 0.814932},
   "outage": {"type": "noul", "noul": 0.86562}},
 "usage": {"input_tokens": 300, "output_tokens": 0, "cost": 0}}
```

`input_tokens` is the length of the rendered record, every question in one
sequence.

## What the engine computes

Each step follows the checkpoint's own `joint_schema_model.py` (revision
2f3de3dd, SHA-256 0e304cf7...), which is the specification:

| step | reference | in colibri |
|---|---|---|
| record | `systemone` | the gateway sends the record in its raw form (`CAPS decide_record=raw`): the caller's values, JSON with sorted keys, nothing defaulted |
| rendering | `encode_record`, `question_options`, `render` | `clef_head.h` `clef_render`: the system prompt, `STATE:`, then per question `FIELD`, `ID`, `TYPE`, `INSTRUCTION` (the question id when there are none) and one `OPTION k:` per option as `{"description":...,"option_id":...}`; noul `true` then `false` with the default descriptions, choice labels sorted, score levels in order; every piece tokenized on its own, the state cut from its end to fit `max_length` 16384 |
| tokenizer | `tokenizer.json`, transformers' `Qwen2Tokenizer` | qwen36's BPE with the exact Unicode classes and NFC for a Clef checkpoint; the regex is the one the checkpoint applies, where a combining mark is not part of a letter run |
| backbone | `Qwen3_5Model` (text path) | qwen36's prefill over the whole record, keeping the final-normed hidden state of every position |
| head | `JointSchemaHead.forward` | `clef_head_forward`: LayerNorm over the hidden states, span means for the instructions and the options, lexical option vectors from the LM head's rows (read at the container's precision, not the int8 copy), two evidence routing layers (options attending to the whole sequence), the option summary per question, four decoder layers over the fields, then `prior + sigmoid(gate) * (scale * cosine + residual)` |
| answer | `systemone` | a softmax per question; the gateway shapes the Jev reply (`choice`, `score` as the expected level, `noul` as P(true)) |

The head runs in f32 (512 MB). The backbone runs at one of three widths,
`COLI_DENSE_BITS`:

| width | RAM (measured RSS) | accuracy vs the reference | speed |
|---|---|---|---|
| f16 (`16`): the container's own values | 55.4 GB | max \|dp\| 0.012 | 47.4 s per request |
| int8 (`8`) | 29 GB | max \|dp\| 0.22 | 20.4 s |
| int4 (`4`) | 19 GB | max \|dp\| 0.22 | 30.2 s |

**Which one runs.** If you set `COLI_DENSE_BITS`, that one. Otherwise `coli
serve`, `coli web` and `coli chat` ask the planner (`coli plan` shows the same
numbers): when its RAM budget holds the trunk at its stored size plus the
runtime and the KV state of 16384 tokens, Clef runs in f16, the width that
answers like the release; when it does not, int8, and the gateway says which on
start-up:

```
[clef] dense trunk in int8: f16 would need 58.9 GiB, the budget is 53.3 GiB (COLI_DENSE_BITS=16 to force it)
```

The planner's margins are its usual ones: on the 61 GB test box f16 ran at
55.4 GB, but the planner keeps it to int8 there. A box with about 72 GB free
gets f16 by default. The engine started by hand (`SERVE=1 ./qwen36`) stays at
int8 unless told otherwise.

## Checked against the reference

- **Tiny oracle** (`make -C c clef-tiny-check`, in CI): a random checkpoint
  in the release's layout (`tools/make_clef_tiny.py`, numpy) and what Clef's
  own code answers on it with transformers 5.10.2 in f32
  (`clef_tiny/ref.json`, written by `tools/make_clef_ref.py`): 16 cases, 44
  questions, every type, 1 to 40 options, null, empty and JSON descriptions,
  noul criteria given, null and absent, JSON instructions and states,
  conversations, added tokens and NFC-decomposed text in the state, the cut
  at `max_length` and a schema that does not fit. Input ids and spans
  identical, 44/44 decisions identical, max |dp| 2.1e-6 (tolerance 2e-5),
  then the same fixture through `coli serve` and both TypeSafe SDKs.
- **The real checkpoint** (`tools/compare_clef.py`, Ryzen 7 8700GE, 8
  threads): 15 requests, 39 questions (the model card's examples, the Jev
  examples of these docs, a contract, a conversation, unicode, JSON, a
  20-intent choice, nine questions in one record) against the release's code
  in bf16, its own precision:

| engine backbone | input ids | decisions | max \|dp\| | requests under 0.01 |
|---|---|---|---|---|
| f16 (`COLI_DENSE_BITS=16`) | 15/15 identical | 39/39 | 0.012 | 14/15 |
| int8 | 15/15 identical | 39/39 | 0.22 | 8/15 |
| int4 (`COLI_DENSE_BITS=4`) | 15/15 identical | 39/39 | 0.22 | 9/15 |

  The gap is the int8 trunk, not the rendering or the head: with the same
  rendering, the same head code and the container's f16 weights the largest
  difference falls from 0.22 to 0.012 (the nine-question invoice record), and
  the code-review verdict that moved by 0.22 in int8 moves by 0.0008. What
  is left at f16 is the reference computing in bf16 where the engine computes
  in f32. On the tiny fixture, where both sides run f32, f16 weights give
  max |dp| 2.5e-6 and int8 5.0e-2.

## Speed

Measured on the same box, 8 threads, every request from a cold state:

| | per request |
|---|---|
| engine, f16, median over the 15 requests (150 to 779 tokens) | 47.4 s |
| engine, int8, the same requests | 20.4 s |
| engine, int4, the same requests | 30.2 s |
| reference, bf16, transformers 5.17 on the CPU, 28 of 64 layers read from disk | 10.8 s |
| `coli serve`, int8, the request above (300 tokens), at the client | 20.6 s, of which 1.4 ms gateway |

The time is the backbone's prefill; the reference's bf16 matrix kernels are
faster on this CPU than qwen36's int8 prefill, int4 is slower because the
prefill is compute-bound, and f16 multiplies in f32. The engine's own time is
in the `x-colibri-engine-ms` header.

## Knobs

| variable | default | meaning |
|---|---|---|
| `COLI_DENSE_BITS` | f16 if the planner's budget holds it, else int8 (see above) | `16`, `8` or `4`; `COLI_DENSE_INT4` as for Qwen3.8-27B ([qwen36.md](qwen36.md#the-dense-27b)) |
| `COLI_CLEF_MAX_LEN` | 16384 | the reference's `max_length`: the state is cut to fit it, a schema longer than it is a 422 |
| `Q36_MAXT` | 16384 for a Clef checkpoint (8192 for the others) | the engine's context. The KV rows are allocated as a record needs them, 2 GiB at 16384 on the 27B, and `coli plan` prices them at 16384. A lower `Q36_MAXT` refuses a longer record with a 422 that names it |

## Limits

- **No pictures.** Clef reads images and video, but only through its Python
  `systemone()`, as PIL images; Jev's `/v1/systemone` request (both official
  SDKs) has no image field, so neither does colibri's.
- **Every request reads the whole record.** There is no snapshot of a state
  across requests yet: a second request about the same document pays its
  prefill again.
- **Clef-Flash** (Qwen3.5-9B backbone, same head code) has the shape this
  path expects but has not been run here.
- The chat of the same process is Qwen3.8-27B's with Clef's post-trained
  layers. It works (asked in English for the capital of France, it answered
  "the capital of France is Paris" in Chinese, after a reasoning block in
  Chinese); its quality has not been measured.
