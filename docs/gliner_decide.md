# GLiNER2.5-Decide on colibri

[GLiNER2.5-Decide](https://huggingface.co/fastino/GLiNER2.5-Decide), by
fastino (Apache-2.0), is a classifier for operational decisions: intent,
routing, sentiment, urgency, moderation, yes/no questions over a passage,
ordinal scores. It is a GLiNER2 checkpoint, a DeBERTa-v3-large encoder with
GLiNER2's heads, and it never generates text.

`c/gliner_decide` runs it from the official checkpoint with no conversion,
and `coli serve` exposes it on `POST /v1/systemone`, the same request and
reply as TypeSafe's Jev API and as every other model colibri serves
([systemone.md](systemone.md#decision-engines)).

## Download and run

The checkpoint is the root of the repository (1.9 GB, F32):

```sh
hf download fastino/GLiNER2.5-Decide model.safetensors config.json \
   "encoder_config/*" tokenizer.json tokenizer_config.json special_tokens_map.json \
   --local-dir ~/Models/GLiNER2.5-Decide
make -C c gliner_decide
./c/coli serve --model ~/Models/GLiNER2.5-Decide
```

`coli` recognises the checkpoint by its `config.json` (`model_type`
`extractor`, GLiNER2's) and the encoder's `encoder_config/config.json`; the
registry keys it `gliner2_span_deberta-v2`. `coli info` shows the encoder and
the token budget, `coli plan` the RAM, `coli doctor` what is missing.

```sh
curl -s localhost:8000/v1/systemone -H 'Content-Type: application/json' -d '{
  "model": "jev-latest",
  "state": "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan.",
  "questions": {
    "department": {"type": "choice", "instructions": "Which department should handle this?",
                   "criteria": {"billing": "invoices, payments, refunds",
                                "technical": "bugs, outages, system errors", "other": "everything else"}},
    "urgency": {"type": "score", "instructions": "How urgent is this?",
                "criteria": ["not urgent", "soon", "blocking"]},
    "churn_risk": {"type": "noul", "instructions": "Does the user threaten to cancel or leave?"}}}'
```

```json
{"id": "req_...", "model": "gliner2.5-decide", "provider": "colibri",
 "answers": {
   "department": {"type": "choice", "choice": "billing",
                  "probabilities": {"billing": 0.751237, "technical": 0.092047, "other": 0.156716},
                  "confidence": 0.626855},
   "urgency": {"type": "score", "score": 1.218665,
               "legend": {"0": "not urgent", "1": "soon", "2": "blocking"},
               "probabilities": {"0": 0.167397, "1": 0.446541, "2": 0.386062}, "confidence": 0.169812},
   "churn_risk": {"type": "noul", "noul": 0.995977}},
 "usage": {"input_tokens": 118, "output_tokens": 0, "cost": 0}}
```

That is the reply `coli serve` gave on the release (the `id` shortened).
`input_tokens` is the length of the one sequence the encoder read. Chat,
completions and messages answer 400 with a pointer to `/v1/systemone`, and
`coli chat` says the same before starting anything.

## From System One to the model's task

GLiNER2 has no notion of choice, score or noul: it classifies a text against
named tasks, each with its labels. The engine turns a System One request into
the call the model card makes, `classify_text(state, tasks)`, one task per
question, in the order given:

| System One | GLiNER2 task |
|---|---|
| question id | the task name (`department`, `urgency`) |
| `instructions` | the task's `prompt` (`department: Which department should handle this?`) |
| `choice` criteria | the labels, in order; a non-empty description becomes a label description |
| `score` criteria | the labels `"0"` .. `"n-1"`, the way the model card writes an ordinal scale; each level's text becomes its description |
| `noul` | the labels `"yes"` then `"no"`, the order of every yes/no example of the model card; `criteria.true` / `criteria.false` describe them |

Every task is single-label, so its probabilities are the softmax over its
labels, which is what `classify_text` returns for a single-label task. The
reply follows from them as on every engine: `choice` is the argmax, `score`
the expected level, `noul` the probability of `yes`.

Two consequences of reading the request as one call:

- **All questions share one sequence.** The encoder reads every question's
  schema and the state together, so a question's probabilities depend on the
  other questions of the same request, as they do in the package when one
  call scores several tasks (the model card's email triage). One request is
  one forward pass, whatever the number of questions.
- **Missing instructions are filled by the gateway** with the text the
  language-model path asks with (`Which of the following applies?`,
  `Rate this on the scale below.`, `Is this true?`). For GLiNER2 that text
  becomes the task's prompt. Measured on eleven single-task examples of the
  model card, asked with no prompt and with the default one (`Is this true?`
  for the yes/no labels, `Rate this on the scale below.` for the 0-5 and 0-10
  scales, `Which of the following applies?` for the rest), the default prompt
  changed no decision and moved the top probability by less than 0.001 up to
  0.12 (refund_request 0.997 -> 0.969, seat_change 0.501 -> 0.622). Send
  `instructions` when you have them.

## What the engine computes

Each step follows the `gliner2` package (2.0.0) and the transformers
DeBERTa-v2 code it runs, which are the reference:

| step | reference | in `c/gliner_decide.c` |
|---|---|---|
| the text | `processor._collate_batch` | a `.` appended when the state does not end in `.`, `!` or `?`; an empty state is `.` |
| words | `processing/word_splitter.WhitespaceTokenSplitter` | Python's regex (URLs, e-mails, @handles, `\w+(?:[-_]\w+)*`, `\S`) with Python's own `\w`, `\s` and `re.IGNORECASE`, each word lower-cased with `str.lower()` (final sigma and dotted capital I included); the tables come from Python by `tools/gen_gliner_unicode.py` |
| schema | `Schema.classification`, `processor._transform_schema` | per task `( [P] name[: prompt][ [DESCRIPTION] label: text ...] ( [L] label ... ) )`, tasks joined by `[SEP_STRUCT]`, then `[SEP_TEXT]` and the words |
| tokens | `processor._format_input_with_mapping` | every element above tokenized on its own, no `[CLS]` or `[SEP]`; the classifier reads the first token of each `[L]` |
| tokenizer | `tokenizer.json` (DebertaV2TokenizerFast) | added tokens split out of the raw text, `Replace(\s{2,}\|[\n\r\t] -> " ")`, NFC, right strip, `[UNK]` matched after normalization, Metaspace (prepend always, split), Unigram (best-scoring segmentation, unknown characters at min score - 10, adjacent unknowns fused into one `[UNK]`) |
| encoder | `DebertaV2Model` | word embeddings and LayerNorm (no absolute positions); 24 post-norm layers of disentangled attention: `(q.k + q.pos_k[r] + k.pos_q[r]) / sqrt(3 * 64)` with `r` the relative position in log buckets (256 buckets, maximum 512, clamped to the 512-row table), `pos_q` and `pos_k` the LayerNorm'd relative embeddings through the layer's own query and key projections; GELU FFN |
| head | `runtime._extract_classification_result` | the classifier (Linear 1024 -> 2048, ReLU, Linear 2048 -> 1) at each `[L]`, temperature 1, softmax over the task's labels |

The span, count and relation heads (`span_rep`, `count_pred`,
`count_embed`) serve extraction and are not loaded. The relative embeddings
are the same for every request, so their projections are computed once at
load. Weights are kept in f32 as released.

## Checked against the reference

- **Tiny oracle** (`make -C c gliner-decide-tiny-check`, in CI): a random
  checkpoint in the release's layout (`tools/make_gliner_decide_tiny.py`,
  numpy, a Unigram tokenizer counted from a small corpus, a head size of 24,
  16 position buckets with a maximum of 64 so every sequence reaches the log
  buckets and the clamp) and what `gliner2` answers on it
  (`gliner_decide_tiny/ref.json`, written by `tools/make_gliner_decide_ref.py`):
  13 requests with 26 questions plus one whose questions do not fit, covering
  every type, 2 to 40 options, labels with and without descriptions, missing
  instructions, long states and a conversation cut at `max_len`, JSON states
  and instructions, the GLiNER2 marker tokens inside the text, URLs, e-mails,
  handles, unicode and an empty state; plus 30 tokenizer strings and 15
  word-splitter strings. Token ids and markers identical, probabilities within
  2.0e-6 (tolerance 2e-5), every decision the same; then the same fixture
  through `coli serve`.
- **Tokenizer and word splitter fuzzing** (during development, not in CI):
  20,013 random strings per seed mixing ASCII, whitespace of every kind, Greek
  (final sigma), Cyrillic, CJK, combining marks, emoji, the Kelvin sign, long
  s, dotted and dotless i, URLs, e-mails and the added tokens. Four seeds
  against the release's tokenizer and the word splitter, two against the tiny
  tokenizer: 0 differences in token ids, 0 in words.
- **The real checkpoint** (`tools/compare_gliner_decide.py`): 39 requests, 79
  questions: every example of the model card written as System One questions
  (the multi-label ones as one noul per label), the Jev examples of
  systemone.md, a contract of about 960 tokens, a conversation, JSON states
  and instructions, 40 banking intents, ten questions on one ticket, unicode,
  emoji, URLs, the marker tokens in the text, an empty state. Token ids
  identical on 39/39, decisions 79/79 the same, max |dp| 1.7e-6.

## Speed

Measured on an i7-1355U laptop (10 cores, 12 threads), 6 threads for both
sides, on a machine shared with other jobs: the load average stayed between
10 and 13 during the runs below, so absolute numbers move from run to run.
Engine and package were run in turn, request by request, so both saw the
same load (median of 5 each):

| request | tokens | `coli serve`, at the client | of which engine | `gliner2` package (torch, CPU, f32) |
|---|---|---|---|---|
| one question (model card, spam or not) | 37 | 294 ms | 290 ms | 718 ms |
| the README request, 3 questions | 118 | 897 ms | 888 ms | 1217 ms |
| ten questions on one ticket | 286 | 1955 ms | 1950 ms | 2231 ms |
| a contract, 3 questions | 965 | 8401 ms | 8397 ms | 9457 ms |

The gateway's share is 4 to 10 ms. Over the 39 requests of the comparison
the engine's median was 502 ms per request (load average 5 to 8 when
sampled). A request is one forward pass whatever the number of questions,
so its time follows the length of the sequence: the ten questions above are
one pass over 286 tokens, most of them the questions' own schema.

Longer sequences cost more than in proportion, the attention growing with
the square of the length. A three-label question over a repeated support
text, same machine: 0.54 s at 81 tokens and 2.8 s at 494 with the machine
otherwise idle; with a load average of 7 to 15, 0.69 s at 81, 4.0 s at 494,
8.7 s at 970, 25.7 s at 1923 and 93 s at 3829. `COLI_GLINER_MAX_LEN` bounds
it.

RAM: 1.81 GB resident for the engine serving the release (peak 1.88 GB):
the f32 encoder and classifier, 436M parameters of the 486M in the file.

## Knobs

| variable | default | meaning |
|---|---|---|
| `COLI_GLINER_MAX_LEN` | 4096 | tokens of the one sequence a request becomes: the questions first, then as many whole words of the state as fit |
| `OMP_NUM_THREADS` | physical cores | as for the other engines |

Questions whose schema alone exceeds `COLI_GLINER_MAX_LEN` are refused with a
422 that says how many tokens they take.

## Limits

- **English.** The model card measures this checkpoint on English; fastino
  points to GLiNER2.5-multi-Decide for other languages, which this engine has
  not been run on.
- **No calibration ships with the checkpoint.** The probabilities are the
  softmax of the classifier at temperature 1, as the package computes them;
  `temperature` in the DECISION frame is `null`.
- **Single-label only.** System One has no multi-label question, so the
  package's `multi_label` mode (a sigmoid per label and a threshold) is not
  reachable; ask one noul per label instead, as the comparison above does.
- **Long states.** The package reads a whole text by default; the engine
  stops at `COLI_GLINER_MAX_LEN` tokens so that one request cannot take the
  machine, and `state_dropped` in the DECISION frame says how many state
  tokens were left out. Below the limit the two read the same tokens. The
  cost grows with the square of the length (see Speed).
- **The `noul` label order.** The engine asks `yes` before `no`; the order
  moves the probabilities a little (handoff example of the model card:
  p(yes) 0.325 asked yes-first, 0.297 asked no-first), never the decision on
  the examples measured.
- **Question ids that extend each other.** The package finds a task's answer
  by the prefix of its schema text, so with two question ids such as `risk`
  (with instructions `score it`) and `risk: score`, `classify_text` returns
  only one of them (checked on the tiny fixture). The engine answers every
  question by position.
- **Other GLiNER2 checkpoints.** The registry accepts any span GLiNER2
  checkpoint with a DeBERTa-v2/v3 encoder, and a 24 x 1024 encoder is named
  GLiNER2.5-Decide. Only `fastino/GLiNER2.5-Decide` was compared with the
  package; the boundary architecture is refused.
