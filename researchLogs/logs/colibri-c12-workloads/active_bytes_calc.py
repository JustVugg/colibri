#!/usr/bin/env python3
"""Active-bytes-per-token arithmetic for MoE decode, from GGUF tensor tables."""
import math, os

def load(tfile, fs):
    recs = []
    for line in open(tfile):
        line = line.rstrip('\n')
        if not line:
            continue
        name, tt, dims, ofs = line.split('\t')
        recs.append((name, int(tt), dims, int(ofs)))
    recs.sort(key=lambda r: r[3])
    size = {}
    for i, (n, t, d, o) in enumerate(recs):
        end = recs[i + 1][3] if i + 1 < len(recs) else fs
        size[n] = end - o
    return recs, size

FILES = [
    ("35B-IQ2", "/Users/creemac/.lmstudio/models/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf",
     "phaseI3_iq2xxs_tensors.txt", 40, 256, 8),
    ("35B-IQ3", "/Users/creemac/.lmstudio/models/unsloth/Qwen3.6-35B-A3B-UD-IQ3_XXS/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf",
     "phaseI3_iq3xxs_tensors.txt", 40, 256, 8),
    ("80B-IQ2", "/Users/creemac/.lmstudio/models/unsloth/Qwen3-Next-80B-A3B-Instruct-UD-IQ2_XXS/Qwen3-Next-80B-A3B-Instruct-UD-IQ2_XXS.gguf",
     "phaseB80_audit_tensors.txt", 48, 512, 10),
]
BASE = "researchLogs/logs/colibri-c12-workloads/"

for tag, gguf, tfile, layers, n_exp, topk in FILES:
    fs = os.path.getsize(gguf)
    recs, size = load(BASE + tfile, fs)
    shared = sum(size[n] / layers for n, t, d, o in recs
                 if not n.endswith('_exps.weight') and '.nextn.' not in n)
    expert_pool = sum(size[n] / layers for n, t, d, o in recs if n.endswith('_exps.weight'))
    act = shared + expert_pool * topk / n_exp
    # shexp info
    shexp = sorted((n, d, f"{size[n]/2**20:.2f}MB") for n, t, d, o in recs
                   if 'shexp' in n and 'gate_inp' not in n)[:3]
    print(f"===== {tag} =====")
    print(f"  file {fs/2**30:.2f} GiB | layers {layers} | experts {n_exp} top-{topk}")
    print(f"  shared/token {shared/2**20:.1f} MB | expert pool/layer {expert_pool/2**20:.1f} MB | active experts {topk}")
    print(f"  ACTIVE BYTES/TOKEN = {act/2**20:.0f} MB")
    print(f"  shexp: {shexp}")
