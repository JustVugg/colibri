"""Populate the Colibri model dir with the full 16-shard weights via hardlinks.

The HF snapshot on D: has all 16 bf16 shards (114 GB incl. AirLLM split
layers). Colibri's dir only had 1 shard. Hardlinking the remaining shards +
tokenizer/config gives Colibri the complete model at zero extra disk cost,
because both paths live on the same volume.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

SNAP = Path(
    r"D:\.cache\huggingface\hub\models--Qwen--Qwen3-Coder-30B-A3B-Instruct\snapshots"
    r"\b2cff646eb4bb1d68355c01b18ae02e7cf42d120"
)
DEST = Path(r"D:\colibri-models\Qwen3-Coder-30B-A3B")

WANTED_GLOBS = [
    "model-000*-of-00016.safetensors",
    "model.safetensors.index.json",
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
    "special_tokens_map.json",
]


def main() -> None:
    if not SNAP.exists():
        raise SystemExit(f"snapshot missing: {SNAP}")
    DEST.mkdir(parents=True, exist_ok=True)

    total = 0
    linked = 0
    skipped = 0
    bytes_linked = 0

    for pattern in WANTED_GLOBS:
        for src in sorted(SNAP.glob(pattern)):
            total += 1
            dst = DEST / src.name
            if dst.exists() and dst.stat().st_size == src.stat().st_size:
                skipped += 1
                continue
            try:
                if dst.exists():
                    dst.unlink()
                os.link(src, dst)  # hardlink: no extra disk
                linked += 1
                bytes_linked += src.stat().st_size
                print(f"linked {src.name} ({src.stat().st_size / 1e9:.2f} GB)")
            except OSError as exc:
                print(f"link failed for {src.name} ({exc}); copying")
                shutil.copy2(src, dst)
                linked += 1
                bytes_linked += src.stat().st_size

    shards = list(DEST.glob("model-000*-of-00016.safetensors"))
    size = sum(p.stat().st_size for p in shards)
    print(f"\ntotal candidates={total} linked={linked} skipped={skipped}")
    print(f"shards present={len(shards)}  shard bytes={size / 1e9:.1f} GB")
    print(f"linking saved {bytes_linked / 1e9:.1f} GB of duplication")
    du = shutil.disk_usage("D:/")
    print(f"D: free {du.free / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
