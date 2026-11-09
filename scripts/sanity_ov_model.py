"""Isolate whether the degenerate NPU output is the device or the model.

Runs the same prompt on CPU and NPU with the chat template, writing UTF-8
output to files so console code pages cannot garble it.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import openvino_genai as g

MODEL = (
    r"D:\.cache\huggingface\models--OpenVINO--Qwen3-4B-int4-ov\snapshots"
    r"\b467368d16b75df14055562fe927ae8e1f15f7ef"
)
OUT = Path(r"D:\Colibri\results")
OUT.mkdir(parents=True, exist_ok=True)

MSGS = [{"role": "user", "content": "Write a Python function add(a, b) that returns a + b. Only the code, no explanation."}]


def run(device: str, max_new_tokens: int = 48) -> str:
    t0 = time.time()
    pipe = g.LLMPipeline(MODEL, device)
    load = time.time() - t0
    cfg = g.GenerationConfig()
    cfg.max_new_tokens = max_new_tokens
    cfg.do_sample = False
    t1 = time.time()
    history = g.ChatHistory()
    for m in MSGS:
        history.append({"role": m["role"], "content": m["content"]})
    res = pipe.generate(history, cfg)
    gen = time.time() - t1
    txt = res.texts[0] if hasattr(res, "texts") else str(res)
    body = f"[{device}] load={load:.1f}s gen={gen:.1f}s chars={len(txt)}\n{txt}\n"
    print(body, flush=True)
    with (OUT / f"sanity_{device.replace('.', '_')}.txt").open("w", encoding="utf-8") as fh:
        fh.write(body)
    return txt


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "CPU"
    run(target)
