"""PyTorch float64 cross-check for the toy QLoRA trainer gradients.

  TRAIN_REF_DUMP=/tmp/fix.json ./tests/test_train_linear
  python3 tools/ref_train_linear.py /tmp/fix.json

Rebuilds y = Wd x + (alpha/rank) * B(A x) with torch autograd in float64 on the
DEQUANTIZED base weights (Wd is exactly what the int4 kernel encodes) and
compares dA/dB/dx with the gradients the C trainer computed. Not part of make
check (torch is not a repo dependency) — run manually when touching train/. """
import json, sys

import torch

fix = json.load(open(sys.argv[1]))
S, O, I, R = fix["S"], fix["O"], fix["I"], fix["rank"]
scale = fix["alpha"] / R
t = lambda k, shape: torch.tensor(fix[k], dtype=torch.float64).reshape(shape)

Wd = t("Wd", (O, I))
A = t("A", (R, I)).requires_grad_()
B = t("B", (O, R)).requires_grad_()
x = t("x", (S, I)).requires_grad_()
tgt = t("tgt", (S, O))

y = x @ Wd.T + scale * (x @ A.T) @ B.T
loss = ((y - tgt) ** 2).mean()
loss.backward()

fails = 0
for name, got, ref in (("dA", t("dA", (R, I)), A.grad),
                       ("dB", t("dB", (O, R)), B.grad),
                       ("dx", t("dx", (S, I)), x.grad)):
    denom = ref.abs().clamp_min(1e-9)
    err = ((got - ref).abs() / denom).max().item()
    ok = err < 5e-4
    fails += not ok
    print(f"{name}: max relerr vs torch float64 = {err:.3e} {'ok' if ok else 'FAIL'}")
sys.exit(1 if fails else 0)
