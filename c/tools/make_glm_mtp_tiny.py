#!/usr/bin/env python3
"""Give a GLM-5.2 tiny fixture (make_glm_oracle.py's glm_tiny) an MTP head, for the
engine's speculative decode in tests.

colibri.c turns MTP on when layer n_layers carries the head's tensors (eh_proj, enorm,
hnorm, shared_head.norm and a full attention + MoE block). The oracle's transformers
model has none, so this writes them: the block is a copy of the last layer's tensors,
eh_proj passes the normalized hidden state through (its embedding half is zero), the
three norms are ones and the model's final norm. The drafts it makes are a model of
the model, not trained ones: some are accepted, most are not, and both paths run.

Greedy decode with speculation verifies every draft, so the tokens are the oracle's
whatever the head proposes: the fixture tests the draft/verify machinery, not the head.

usage: make_glm_mtp_tiny.py --src glm_tiny --out glm_tiny_mtp
"""
import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", default="glm_tiny")
    ap.add_argument("--out", default="glm_tiny_mtp")
    args = ap.parse_args()
    src, out = Path(args.src), Path(args.out)
    cfg = json.loads((src / "config.json").read_text())
    L, D = cfg["num_hidden_layers"], cfg["hidden_size"]
    sd = load_file(str(src / "model.safetensors"))
    last = f"model.layers.{L - 1}."
    head = f"model.layers.{L}."
    for name, t in list(sd.items()):
        if name.startswith(last) and ".indexer." not in name:
            sd[head + name[len(last):]] = t.clone()
    eh = torch.zeros(D, 2 * D, dtype=torch.float32)
    eh[:, D:] = torch.eye(D)
    sd[head + "eh_proj.weight"] = eh
    sd[head + "enorm.weight"] = torch.ones(D)
    sd[head + "hnorm.weight"] = torch.ones(D)
    sd[head + "shared_head.norm.weight"] = sd["model.norm.weight"].clone()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    save_file({k: v.contiguous() for k, v in sd.items()}, str(out / "model.safetensors"))
    for f in src.iterdir():
        if f.name != "model.safetensors" and f.is_file():
            shutil.copy(f, out / f.name)
    print(f"wrote {out}: layer {L} is the MTP head ({sum(1 for k in sd if k.startswith(head))} tensors)")


if __name__ == "__main__":
    main()
