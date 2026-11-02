"""Tokenize an SFT dataset into the coli-sft-v1 format read by train/dataset.h.

Input: JSONL, one sample per line, either
  {"messages": [{"role": "user"|"assistant"|"system", "content": "..."}, ...]}
  {"prompt": "...", "completion": "..."}          (wrapped as user/assistant)
Output in --output dir:
  train.bin/.msk/.idx  valid.bin/.msk/.idx  metadata.json
  .bin = uint32 ids | .msk = uint8 loss mask (1 only on assistant tokens) |
  .idx = int64 {magic, n_samples, cum-offsets[n+1]}

Tokenizer: --model accepts a HF repo id (zai-org/GLM-5.2 — tokenizer files are
a few MB, no weights needed) or a local dir. Prompt tokens are masked out of
the loss; only assistant-turn tokens train (§14 prompt masking).

  python3 tools/prepare_sft.py --input data.jsonl --output data/tokenized \
      --model zai-org/GLM-5.2 --valid-frac 0.05 --seed 0

--synthetic N V: skip tokenization entirely and emit N random samples over a
V-token vocab (deterministic) — for exercising the C reader/trainer against the
tiny oracle model, which has no real tokenizer."""
import argparse, json, os, struct

import numpy as np

MAGIC = 0x434F4C4953465431

ap = argparse.ArgumentParser()
ap.add_argument("--input")
ap.add_argument("--output", required=True)
ap.add_argument("--model", help="HF repo id or local dir with tokenizer files")
ap.add_argument("--valid-frac", type=float, default=0.05)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--max-len", type=int, default=4096, help="truncate samples beyond this")
ap.add_argument("--synthetic", nargs=2, type=int, metavar=("N", "VOCAB"),
                help="emit N random samples over VOCAB ids instead of tokenizing")
args = ap.parse_args()

rng = np.random.default_rng(args.seed)
samples = []          # list of (ids: list[int], mask: list[int])

if args.synthetic:
    n, vocab = args.synthetic
    for _ in range(n):
        L = int(rng.integers(8, 65))
        ids = rng.integers(0, vocab, L).tolist()
        cut = int(rng.integers(1, L))              # fake prompt/completion split
        samples.append((ids, [0]*cut + [1]*(L-cut)))
    tok_name = f"synthetic(vocab={vocab})"
else:
    if not args.input or not args.model:
        raise SystemExit("--input and --model required (or use --synthetic)")
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(args.model)
    tok_name = args.model
    with open(args.input) as f:
        for ln, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if "messages" in d:
                msgs = d["messages"]
            else:
                msgs = [{"role": "user", "content": d["prompt"]},
                        {"role": "assistant", "content": d["completion"]}]
            ids, mask = [], []
            # tokenize turn by turn so the mask lands exactly on assistant tokens
            for i, m in enumerate(msgs):
                upto = tk.apply_chat_template(msgs[:i+1], tokenize=True,
                                              add_generation_prompt=False)
                if not isinstance(upto, list):        # transformers v5 BatchEncoding
                    upto = upto["input_ids"]
                if upto[:len(ids)] != ids:
                    raise SystemExit(f"line {ln}: chat template is not prefix-stable, "
                                     "cannot mask incrementally")
                new = upto[len(ids):]
                ids = upto
                mask += [1 if m["role"] == "assistant" else 0] * len(new)
            ids, mask = ids[:args.max_len], mask[:args.max_len]
            if len(ids) >= 2 and any(mask):
                samples.append((ids, mask))
    if not samples:
        raise SystemExit("no usable samples")

order = rng.permutation(len(samples))
nv = max(1, int(len(samples) * args.valid_frac)) if len(samples) > 1 else 0
splits = {"valid": [samples[i] for i in order[:nv]],
          "train": [samples[i] for i in order[nv:]]}

os.makedirs(args.output, exist_ok=True)
for split, ss in splits.items():
    if not ss:
        continue
    ids = np.concatenate([np.asarray(s[0], dtype=np.uint32) for s in ss])
    msk = np.concatenate([np.asarray(s[1], dtype=np.uint8) for s in ss])
    off = np.zeros(len(ss)+1, dtype=np.int64)
    np.cumsum([len(s[0]) for s in ss], out=off[1:])
    open(os.path.join(args.output, split+".bin"), "wb").write(ids.tobytes())
    open(os.path.join(args.output, split+".msk"), "wb").write(msk.tobytes())
    with open(os.path.join(args.output, split+".idx"), "wb") as f:
        f.write(struct.pack("<qq", MAGIC, len(ss)))
        f.write(off.tobytes())

meta = {"format": "coli-sft-v1", "tokenizer": tok_name, "seed": args.seed,
        "samples": {k: len(v) for k, v in splits.items()},
        "tokens": {k: int(sum(len(s[0]) for s in v)) for k, v in splits.items()}}
json.dump(meta, open(os.path.join(args.output, "metadata.json"), "w"), indent=1)
print(f"{args.output}: " + " | ".join(f"{k}: {len(v)} samples, {meta['tokens'][k]} tokens"
                                      for k, v in splits.items() if v))
