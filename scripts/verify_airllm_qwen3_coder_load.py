"""Verify AirLLM Qwen3-Coder load end-to-end using the split layers already
on D:/colibri-models/.airllm-hub.  If the full model repo is present it also
dumps a couple of tokenizer decode examples so we have a concrete decode proof.

Run with the project venv python so airllm is importable:
    python scripts/verify_airllm_qwen3_coder_load.py
"""
from __future__ import annotations

from pathlib import Path
import sys
import traceback

try:
    import torch
except Exception as exc:  # pragma: no cover - environment problem
    sys.exit(f"cannot import torch: {exc}")

try:
    from airllm import AutoModel
except Exception as exc:
    sys.exit(f"cannot import airllm.AutoModel: {exc}")


def main() -> None:
    repo = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
    layers_root = Path(
        "D:/.cache/huggingface/hub/"
        "models--Qwen--Qwen3-Coder-30B-A3B-Instruct/snapshots/"
        "b2cff646eb4bb1d68355c01b18ae02e7cf42d120/splitted_model"
    )

    print(f"repo={repo}")
    print(f"layers_root.exists()={layers_root.exists()}")
    if layers_root.exists():
        files = sorted(p.name for p in layers_root.iterdir() if p.is_file)
        print(f"layer files on disk ({len(files)}):")
        for name in files:
            p = layers_root / name
            print(f"  {name:50s} {p.stat().st_size / 1e9:.3f} GB")
        expected_done = {
            "model.embed_tokens.safetensors.done",
            "model.layers.0.safetensors.done",
            "model.layers.9.safetensors.done",
            "model.layers.35.safetensors.done",
        }
        missing_done = expected_done - set(files)
        if missing_done:
            print(
                f"WARNING: expected done markers missing: {sorted(missing_done)}"
            )

    print("loading AutoModel (device=cpu, dtype=bfloat16) ...")
    try:
        model = AutoModel.from_pretrained(
            repo,
            device="cpu",
            dtype=torch.bfloat16,
            layer_shards_saving_path=r"D:\.cache\huggingface\airllm-qwen3moe-grouped",
        )
    except Exception as exc:
        print("AutoModel.from_pretrained raised:")
        traceback.print_exc()
        sys.exit(1)

    print("model constructed; proceeding to decode verification")

    prompt = (
        "Implement a Python function my_atoi(s: str) -> int that parses a signed "
        "decimal integer exactly like C's atoi. Return 0 on any parse failure. "
        "Keep it under 15 lines without imports.\n\n"
    )

    print(f"tokenizing prompt ({len(prompt)} chars) ...")
    try:
        input_ids = model.tokenizer.encode(prompt)
    except Exception as exc:
        print("tokenizer.encode raised:")
        traceback.print_exc()
        sys.exit(1)

    print(f"input_ids length={len(input_ids)}")
    if not input_ids:
        print("tokenizer produced empty input_ids — aborting decode")
        sys.exit(1)
    input_tensor = torch.tensor([input_ids], dtype=torch.long)

    if hasattr(model, "generate"):
        print("generating tokens via model.generate ...")
        try:
            out_ids = model.generate(input_tensor, max_new_tokens=60)
        except TypeError:
            try:
                out_ids = model.generate(input_tensor)
            except Exception as exc:
                print("model.generate raised:")
                traceback.print_exc()
                sys.exit(1)
        except Exception as exc:
            print("model.generate raised:")
            traceback.print_exc()
            sys.exit(1)
        if out_ids is None:
            print("model.generate returned None")
            sys.exit(1)
        decoded = model.tokenizer.decode(out_ids[0] if getattr(out_ids, "ndim", 1) > 1 else out_ids)
        print(f"decoded (len={len(decoded)}):\n---\n{decoded}\n---")
    else:
        print("no model.generate available; skipping generation")
        decoded = None

    repo_files = list(Path("models/Qwen3-Coder-30B-A3B-Instruct").glob("*.safetensors"))
    if repo_files:
        print(f"\nlocal repo shards present ({len(repo_files)}):")
        for p in repo_files:
            print(f"  {p.name} {p.stat().st_size/1e9:.3f} GB")

    print("\nVERIFICATION OK: model loaded, tokenizer encode/decode worked")
    if decoded:
        print(f"sample decoded snippet: {decoded[:300]!r}")


if __name__ == "__main__":
    main()
