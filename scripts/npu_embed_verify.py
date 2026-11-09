"""Verify bge-base-en-v1.5-int8 embeddings on the Intel NPU vs CPU.

The NPU handles single-forward-pass models correctly (unlike autoregressive
LLM generation, which miscompiles). This proves the NPU produces usable
embeddings: NPU/CPU cosine similarity ~1.0 and sane semantic geometry.

Run: .venv/Scripts/python.exe scripts/npu_embed_verify.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import openvino_genai as g

MODEL = (
    r"D:\.cache\huggingface\hub\models--OpenVINO--bge-base-en-v1.5-int8-ov"
    r"\snapshots\9a76c2f2cf23bb2fd4a20469a5170cb7ebe0d5d2"
)
OUT = Path(r"D:\Colibri\results\npu_embed_verify.json")

SENTENCES = [
    "The cat sits on the mat.",
    "A feline is resting on a rug.",
    "Python is a programming language.",
    "The stock market fell sharply today.",
    "def add(a, b): return a + b",
]


def cos(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def embed(device):
    t0 = time.time()
    pipe = g.TextEmbeddingPipeline(MODEL, device)
    load = time.time() - t0
    t1 = time.time()
    vecs = np.asarray([np.asarray(pipe.embed_documents([s])[0]) for s in SENTENCES])
    gen = time.time() - t1
    return vecs, load, gen


def main():
    report = {"model": "bge-base-en-v1.5-int8-ov", "sentences": SENTENCES}
    cpu, cpu_load, cpu_gen = embed("CPU")
    report["cpu"] = {"load_s": round(cpu_load, 2), "embed_s": round(cpu_gen, 3),
                     "dim": int(cpu.shape[1])}
    print(f"CPU: load={cpu_load:.1f}s embed={cpu_gen:.3f}s dim={cpu.shape[1]}", flush=True)
    try:
        npu, npu_load, npu_gen = embed("NPU")
        sims = [cos(cpu[i], npu[i]) for i in range(len(SENTENCES))]
        report["npu"] = {"load_s": round(npu_load, 2), "embed_s": round(npu_gen, 3),
                         "dim": int(npu.shape[1]),
                         "cosine_vs_cpu": [round(s, 4) for s in sims],
                         "min_cosine_vs_cpu": round(min(sims), 4)}
        print(f"NPU: load={npu_load:.1f}s embed={npu_gen:.3f}s "
              f"min_cosine_vs_cpu={min(sims):.4f}", flush=True)
        # semantic geometry on the NPU vectors: cat/feline closer than cat/market
        sem = {
            "cat~feline": round(cos(npu[0], npu[1]), 4),
            "cat~market": round(cos(npu[0], npu[3]), 4),
            "python~code": round(cos(npu[2], npu[4]), 4),
        }
        report["npu_semantics"] = sem
        report["npu_ok"] = (min(sims) > 0.97 and sem["cat~feline"] > sem["cat~market"])
        print("semantics:", sem, "| NPU_OK:", report["npu_ok"], flush=True)
    except Exception as e:
        report["npu"] = {"error": str(e)}
        report["npu_ok"] = False
        print("NPU failed:", e, flush=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()