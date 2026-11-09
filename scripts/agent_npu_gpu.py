"""Run a real LLM agent on the Intel NPU (and optionally Arc iGPU / RTX 4060).

Proves the NPU is a usable inference target on this laptop, and reports the
real memory budget that OpenVINO and Windows expose for each accelerator.

Usage:
    python scripts/agent_npu_gpu.py --device NPU
    python scripts/agent_npu_gpu.py --device GPU.0     # Intel Arc iGPU
    python scripts/agent_npu_gpu.py --device GPU.1     # RTX 4060 (OpenVINO)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import openvino as ov

OUT = Path(r"D:\Colibri\results\agent_npu_gpu.json")
OUT.parent.mkdir(parents=True, exist_ok=True)

DEFAULT_MODEL = (
    r"D:\.cache\huggingface\models--OpenVINO--Qwen3-4B-int4-ov\snapshots"
    r"\b467368d16b75df14055562fe927ae8e1f15f7ef"
)


def memory_table() -> dict:
    core = ov.Core()
    table = {}
    for dev in ("CPU", "GPU.0", "GPU.1", "NPU"):
        if dev not in core.available_devices:
            continue
        entry = {}
        for prop in ("FULL_DEVICE_NAME", "DEVICE_TYPE", "GPU_DEVICE_TOTAL_MEM_SIZE"):
            try:
                entry[prop] = core.get_property(dev, prop)
            except Exception:
                pass
        table[dev] = {k: str(v) for k, v in entry.items()}
        if "GPU_DEVICE_TOTAL_MEM_SIZE" in entry:
            table[dev]["total_gb"] = round(entry["GPU_DEVICE_TOTAL_MEM_SIZE"] / 1e9, 2)
    return table


def run(device: str, model_dir: str, prompt: str, max_new_tokens: int) -> dict:
    import openvino_genai as ov_genai

    print(f"loading {model_dir} onto {device} ...")
    t0 = time.time()
    pipe = ov_genai.LLMPipeline(model_dir, device)
    load_s = time.time() - t0
    print(f"loaded on {device} in {load_s:.1f}s")

    cfg = ov_genai.GenerationConfig()
    cfg.max_new_tokens = max_new_tokens
    cfg.temperature = 0.2
    cfg.do_sample = False

    t1 = time.time()
    out = pipe.generate(prompt, cfg)
    gen_s = time.time() - t1
    text = out.texts[0] if hasattr(out, "texts") else str(out)
    print(f"\n--- {device} output ({gen_s:.1f}s) ---\n{text}\n---")

    return {
        "device": device,
        "model": model_dir,
        "load_s": round(load_s, 2),
        "generate_s": round(gen_s, 2),
        "chars": len(text),
        "output": text,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="NPU")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument(
        "--prompt",
        default=(
            "You are a coding agent. Write a Python function add(a, b) that returns "
            "the sum, then one pytest test for it. Be brief.\n\n"
        ),
    )
    args = ap.parse_args()

    report = {"memory": memory_table(), "run": None}
    print(json.dumps(report["memory"], indent=2))

    if not Path(args.model).exists():
        raise SystemExit(f"model dir missing: {args.model}")

    report["run"] = run(args.device, args.model, args.prompt, args.max_new_tokens)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
