#!/usr/bin/env python3
"""Compare two K3_LOGITS dumps position by position.

rel_l2 answers "did the numbers move"; top-1 agreement and the top-2 margin
answer "would the model have said something else", which is the only question
that matters for a kernel swap. A change that never flips an argmax is invisible
in generation; one that flips argmax on a narrow-margin position is not, however
small its L2.

usage: logit_cmp.py a.bin b.bin [vocab]
"""
import sys
import numpy as np

a_path, b_path = sys.argv[1], sys.argv[2]
vocab = int(sys.argv[3]) if len(sys.argv) > 3 else 163584

a = np.fromfile(a_path, dtype=np.float32)
b = np.fromfile(b_path, dtype=np.float32)
if a.size != b.size:
    sys.exit(f"size mismatch: {a.size} vs {b.size} floats")
if a.size % vocab:
    sys.exit(f"{a.size} floats is not a multiple of vocab {vocab}")

n = a.size // vocab
a = a.reshape(n, vocab)
b = b.reshape(n, vocab)
print(f"{n} positions, vocab {vocab}")

num = np.linalg.norm(a - b, axis=1)
den = np.linalg.norm(a, axis=1)
rel = np.where(den > 0, num / np.maximum(den, 1e-30), 0.0)

ta, tb = a.argmax(1), b.argmax(1)
flips = np.nonzero(ta != tb)[0]

# Margin between top-1 and top-2 on the reference side: a flip where this is tiny
# is a coin toss the kernel had no real say in.
part = np.partition(a, -2, axis=1)
margin = part[:, -1] - part[:, -2]

print(f"rel_l2 per position : mean {rel.mean():.3e}  max {rel.max():.3e}")
print(f"top-1 agreement     : {n - len(flips)}/{n} ({100.0 * (n - len(flips)) / n:.2f}%)")
print(f"top-2 margin (ref)  : min {margin.min():.4f}  median {np.median(margin):.4f}")
if len(flips):
    print("flipped positions:")
    for i in flips:
        print(f"  pos {i:4d}  ref {ta[i]:6d} -> {tb[i]:6d}  "
              f"ref margin {margin[i]:.5f}  rel_l2 {rel[i]:.3e}")
else:
    print("no argmax flips")
sys.exit(1 if len(flips) else 0)
