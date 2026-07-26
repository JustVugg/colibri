#!/usr/bin/env python3
"""Tiny SFT set for the QLoRA quickstart (scripts/quickstart_tiny.sh).

Builds a coli-sft-v1 dataset for the tiny oracle model (glm_tiny) out of the
oracle's OWN prompt: each sample is ref_glm.json's prompt_ids followed by a
deterministic alternative continuation (loss-masked to the continuation only).
Overfitting a LoRA on this makes greedy generation from the oracle prompt emit
OUR continuation instead of the reference one — while with the adapter off the
engine still matches the reference exactly. No tokenizer needed anywhere.

Usage: python3 tools/make_tiny_sft.py [--ref ref_glm.json] [--out data/tiny_sft]
Writes train.bin/.msk/.idx (+ a copy as valid.*) and target_ids.json.
"""
import json, struct, argparse, os

TDS_MAGIC = 0x434F4C4953465431  # "COLISFT1"
NSAMP = 8                       # identical samples: this is a deliberate overfit

ap = argparse.ArgumentParser()
ap.add_argument("--ref", default="ref_glm.json")
ap.add_argument("--out", default="data/tiny_sft")
args = ap.parse_args()

ref = json.load(open(args.ref))
prompt, full = ref["prompt_ids"], ref["full_ids"]
np_, nfull = len(prompt), len(full)
ncont = nfull - np_
vocab = 256

# deterministic alternative continuation: a simple alternating pattern —
# trivially learnable AND robust to autoregressive drift (any prefix of it
# still predicts the same next token). Nudged wherever it would collide with
# the reference so "reproduces target" vs "matches base" is unambiguous.
target = []
for i in range(ncont):
    t = 42 if i % 2 == 0 else 7
    if t == full[np_ + i]:
        t += 1
    target.append(t)

sample = prompt + target
msk = [0] * np_ + [1] * ncont     # loss on the continuation only

toks, msks, lens = [], [], []
for _ in range(NSAMP):
    toks += sample; msks += msk; lens.append(len(sample))

os.makedirs(args.out, exist_ok=True)
def wr(name, data):
    with open(os.path.join(args.out, name), "wb") as f: f.write(data)

bin_ = struct.pack(f"<{len(toks)}I", *toks)
msk_ = bytes(msks)
idx = struct.pack("<2q", TDS_MAGIC, NSAMP)
cum = 0
idx += struct.pack("<q", cum)
for L in lens:
    cum += L; idx += struct.pack("<q", cum)

for split in ("train", "valid"):
    wr(split + ".bin", bin_); wr(split + ".msk", msk_); wr(split + ".idx", idx)

json.dump({"prompt_ids": prompt, "target_ids": target,
           "ref_continuation": full[np_:]},
          open(os.path.join(args.out, "target_ids.json"), "w"))
json.dump({"format": "coli-sft-v1", "source": "tools/make_tiny_sft.py",
           "samples": NSAMP, "sample_len": len(sample), "prompt_len": np_},
          open(os.path.join(args.out, "metadata.json"), "w"))
print(f"tiny SFT: {NSAMP} samples of {len(sample)} tokens "
      f"({np_} prompt + {ncont} target) -> {args.out}")
print("seq-len to use:", len(sample))
