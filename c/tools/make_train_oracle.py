"""Training oracle for Milestone 3: PyTorch ground truth for the C trainer.

Loads the tiny GLM (c/glm_tiny, made by make_glm_oracle.py), attaches the SAME
LoRA adapter the C engine loads (y += alpha/rank * B(Ax) on self_attn.o_proj),
teacher-forces the ref_glm.json token sequence with next-token cross-entropy,
and records float32 ground truth:
  - loss at step 0;
  - dA/dB for every adapted layer at step 0;
  - the loss trajectory over K AdamW steps (lr 1e-2, betas .9/.999, wd 0);
  - final A/B tensors after K steps.
Everything lands in one JSON fixture consumed by tests/test_train_tiny.c.

  python3 tools/make_train_oracle.py --adapter /tmp/ad --out /tmp/fixture.json

Run make_lora_adapter.py first (--init random) so C and PyTorch start from the
identical adapter. Not part of make check (torch is not a repo dependency)."""
import argparse, json, os

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import GlmMoeDsaForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="glm_tiny")
ap.add_argument("--ref", default="ref_glm.json")
ap.add_argument("--adapter", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--steps", type=int, default=8)
ap.add_argument("--lr", type=float, default=1e-2)
args = ap.parse_args()

torch.manual_seed(0)
model = GlmMoeDsaForCausalLM.from_pretrained(
    args.model, torch_dtype=torch.float32, attn_implementation="eager")
model.eval()
for p in model.parameters():
    p.requires_grad_(False)

meta = json.load(open(os.path.join(args.adapter, "adapter.json")))
rank, alpha = meta["rank"], meta["alpha"]
scale = alpha / rank
sd = load_file(os.path.join(args.adapter, "adapter.safetensors"))

# attach LoRA to o_proj of every adapted layer via forward hooks
params, layers = {}, []
for tgt in meta["targets"]:
    assert tgt.endswith("self_attn.o_proj"), f"only o_proj in M3: {tgt}"
    li = int(tgt.split(".")[2])
    A = sd[tgt + ".lora_A.weight"].clone().requires_grad_()
    B = sd[tgt + ".lora_B.weight"].clone().requires_grad_()
    params[li] = (A, B)
    layers.append(li)
    mod = model.model.layers[li].self_attn.o_proj

    def hook(module, inputs, output, li=li):
        A, B = params[li]
        return output + scale * (inputs[0] @ A.T) @ B.T
    mod.register_forward_hook(hook)

ids = json.load(open(args.ref))["full_ids"]
x = torch.tensor([ids])

def loss_fn():
    logits = model(x).logits.float()
    return F.cross_entropy(logits[0, :-1], x[0, 1:])

# step-0 loss and gradients
loss0 = loss_fn()
loss0.backward()
fix = {"tokens": ids, "rank": rank, "alpha": alpha, "steps": args.steps,
       "lr": args.lr, "layers": layers, "loss0": loss0.item(), "grads": {}, "final": {}}
for li in layers:
    A, B = params[li]
    fix["grads"][str(li)] = {"dA": A.grad.flatten().tolist(),
                             "dB": B.grad.flatten().tolist()}

# K AdamW steps from the same initial state
opt = torch.optim.AdamW([t for li in layers for t in params[li]],
                        lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
opt.zero_grad()
losses = []
for _ in range(args.steps):
    loss = loss_fn()
    losses.append(loss.item())
    opt.zero_grad()
    loss.backward()
    opt.step()
losses.append(loss_fn().item())
fix["losses"] = losses
for li in layers:
    A, B = params[li]
    fix["final"][str(li)] = {"A": A.detach().flatten().tolist(),
                             "B": B.detach().flatten().tolist()}

json.dump(fix, open(args.out, "w"))
print(f"fixture: {args.out} | layers {layers} loss0={loss0.item():.6f} "
      f"trajectory {losses[0]:.4f} -> {losses[-1]:.4f}")
