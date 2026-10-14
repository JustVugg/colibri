# Laya on colibri

[Laya](https://huggingface.co/convaiinnovations/laya), by Convai Innovations
(Apache-2.0), is a decision model: give it a state (text or JSON) and typed
questions (`choice`, `score`, `noul`) and it answers every question with
calibrated probabilities in one forward pass. It never generates text.

`c/laya` runs it from the official checkpoint with no conversion, and
`coli serve` exposes it on `POST /v1/systemone`, the same request and reply
as TypeSafe's Jev API and as every other model colibri serves
([systemone.md](systemone.md#decision-engines)).

## Download and run

The English checkpoint is the root of the repository (842 MB):

```sh
hf download convaiinnovations/laya model.safetensors rl_agent_config.json \
   "encoder/*" "tokenizer/*" --local-dir ~/Models/laya
make -C c laya
./c/coli serve --model ~/Models/laya
```

`coli` recognises the checkpoint by its `rl_agent_config.json` and the
encoder's `encoder/config.json`; there is no root `config.json` to add.
`coli info --model ~/Models/laya` shows the encoder and the token budget,
`coli plan` the RAM.

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
{"id": "req_e622...", "model": "laya", "provider": "colibri",
 "answers": {
   "department": {"type": "choice", "choice": "billing",
                  "probabilities": {"billing": 0.986516, "technical": 0.007975, "other": 0.005509},
                  "confidence": 0.979774},
   "urgency": {"type": "score", "score": 1.772169,
               "legend": {"0": "not urgent", "1": "soon", "2": "blocking"},
               "probabilities": {"0": 0.038904, "1": 0.150022, "2": 0.811074}, "confidence": 0.71661},
   "churn_risk": {"type": "noul", "noul": 0.879019}},
 "usage": {"input_tokens": 164, "output_tokens": 0, "cost": 0}}
```

`input_tokens` is what the encoder read, every question's sequence counted.
Chat, completions and messages answer 400 with a pointer to `/v1/systemone`,
and `coli chat` says the same before starting anything.

## What the engine computes

Each step follows the `laya` package (0.3.24), which is the reference:

| step | reference | in `c/laya.c` |
|---|---|---|
| one sequence per question | `common.build_head`, `build_sequence`, `render_options` | `[CLS] <type> question: <instructions> [SEP] [MASK] option ... [SEP] state [SEP]`; options capped at 48 tokens, shrunk evenly when they overflow `head_max_len`; the state cut at `max_len`, from the left for a list (a conversation keeps its newest turns) |
| tokenizer | `tokenizer/tokenizer.json` | NFC, added tokens split out first, GPT-2 byte-level pre-tokenizer, BPE (`tok.h`) |
| encoder | ModernBERT | 28 pre-norm layers, no norm before layer 0's attention; RoPE theta 160000 on the global layers (every third, from 0) and 10000 on the others, which see +-64 positions; GeGLU MLP; no biases; final LayerNorm |
| head | `DecisionModel.forward` | `+ type_emb[type]`, two norm-first transformer layers (ReLU, 16 heads), the scorer (LayerNorm, Linear, GELU, Linear) at each option's `[MASK]` |
| calibration | `Agent._decode_answers` | one temperature per type and option count (`choice:3-5`, `noul:2`, ...), clamped to [0.5, 5] like the package (`choice:11+` ships 0.1006), softmax over the question's options |
| act head | `DecisionModel.forward` | `[h[0], top1, top1 - top2, entropy, k/255]` -> act / escalate; carried in the DECIDE answer, not in the Jev reply (the model card says it carries no usable signal yet) |

Weights are read as stored (F16) and kept in f32: 1.7 GB of RAM. Every
question of a request is one batch of rows; attention stays inside each
question's sequence.

## Checked against the reference

- **Tiny oracle** (`make -C c laya-tiny-check`, in CI): a random checkpoint
  in the release's layout (`tools/make_laya_tiny.py`, numpy) and what the
  `laya` package answers on it (`laya_tiny/ref.json`, written by
  `tools/make_laya_ref.py`): 12 requests with 25 questions plus one the
  package refuses, covering every type, 2 to 20 options, option shrinking,
  long states and conversations cut at `max_len`, JSON states and
  instructions, `[MASK]` in the text, NFC and the added tokens. Token ids and
  markers identical, probabilities within 4.8e-7 (tolerance 2e-5), every
  decision the same; then the same fixture through `coli serve`.
- **The real checkpoints** (`tools/compare_laya.py`): 36 requests, 117
  questions (the model card's examples, the package's presets, documents cut
  at 512 tokens, a 40-option question, unicode, emoji). Root checkpoint:
  117/117 decisions identical, max |dp| 2.7e-6. `typed-decisions`: 117/117,
  max |dp| 1.2e-6.

## Speed

Measured on an i7-1355U laptop (10 cores), 6 threads for both, with other
processes on the machine, so the numbers move by 20-30% from run to run:

| | `c/laya` | `laya` package (torch, CPU, f32) |
|---|---|---|
| 36 requests, median per request | 1.9 s | 3.1 s |
| one short question (41 tokens) | 300 ms | 250 ms |
| README request, 3 questions, through `coli serve` | 882 ms, of which 2.5 ms gateway | |
| Jev request, 1 question, through `coli serve` | 219 ms, of which 2.5 ms gateway | |

The engine is ahead when a request has several questions and slightly behind
on a single short one. There is no int8 option: measured on the same 117
questions, int8 weights changed 1 decision and int8 weights with int8
activations changed 4, which is not a trade a decision model should make.

## Knobs

| variable | default | meaning |
|---|---|---|
| `COLI_LAYA_MAX_LEN` | the checkpoint's `max_len` (512) | tokens per question sequence; the encoder reads up to 8192 |
| `COLI_LAYA_HEAD_MAX_LEN` | the checkpoint's `head_max_len` (192) | tokens shared by the instructions and the options; raise it for many options |
| `OMP_NUM_THREADS` | physical cores | as for the other engines |

A question whose options do not fit is refused with a 422 that says how many
markers fit, as the package refuses it.

## The other checkpoints

- **`typed-decisions/`** (same architecture, fine-tuned, `max_len` 1024):
  point `--model` at the subfolder. Checked as above.
- **`multilingual/`** (`laya-multilingual`, mmBERT-base, 22 layers x 768,
  256k vocabulary) is not supported yet. What it needs: a SentencePiece-style
  BPE (`Metaspace` pre-tokenizer, a `Replace` normalizer, byte fallback,
  580k merges) in the tokenizer, `<bos>`/`<eos>`/`<mask>` as the special
  tokens, RoPE 160000 on every layer (both thetas in its config), and the
  long-context default (`max_len` 1024, up to 8192). The encoder and the head
  are otherwise the code above; the registry already names it
  (`Laya multilingual`, model id `laya-multilingual`).

## Limits

- English only on this checkpoint: the model card measures it collapsing on
  non-Latin scripts while staying confident.
- The shipped temperatures are fitted on the authors' data; the model card
  advises refitting on yours. colibri applies the checkpoint's own values.
- `noul` can follow its option labels rather than the state on this
  checkpoint (model card); a two-option `choice` with neutral keys is the
  workaround they give.
