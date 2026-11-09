"""Finalize the C: -> D:\\.cache hub merge for Qwen3-Coder-30B-A3B-Instruct.

Verifies every file under the C: source exists at the D: destination with an
identical size and SHA-256, then removes the C: source tree so the duplicate
~109 GB copy does not keep C: full.

Usage:
    python scripts/finalize_hub_merge.py [--delete]

Without --delete it only verifies and reports.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import time
from pathlib import Path

SRC = Path(r"C:\Users\Liso1\.cache\huggingface\hub\models--Qwen--Qwen3-Coder-30B-A3B-Instruct")
DEST = Path(r"D:\.cache\huggingface\hub\models--Qwen--Qwen3-Coder-30B-A3B-Instruct")
LOG = Path(r"D:\Colibri\_finalize_hub_merge.log")


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')}  {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def sha256(path: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--delete", action="store_true", help="remove the C: source after verification")
    args = ap.parse_args()

    log("=" * 70)
    log("FINALIZE START")
    log(f"src  = {SRC}  exists={SRC.exists()}")
    log(f"dest = {DEST}  exists={DEST.exists()}")

    if not SRC.exists():
        log("src already gone -> nothing to merge")
        return 0
    if not DEST.exists():
        log("FATAL: destination missing")
        return 1

    src_files = sorted(p for p in SRC.rglob("*") if p.is_file())
    log(f"src files={len(src_files)}")

    missing: list[str] = []
    sizediff: list[str] = []
    hashdiff: list[str] = []
    verified = 0
    total_bytes = 0
    t0 = time.time()

    for i, sp in enumerate(src_files, 1):
        rel = sp.relative_to(SRC)
        dp = DEST / rel
        if not dp.exists():
            missing.append(str(rel))
            log(f"MISSING {rel}")
            continue
        s_size = sp.stat().st_size
        d_size = dp.stat().st_size
        total_bytes += s_size
        if s_size != d_size:
            sizediff.append(f"{rel} src={s_size} dest={d_size}")
            log(f"SIZE-DIFF {rel} src={s_size} dest={d_size}")
            continue
        s_hash = sha256(sp)
        d_hash = sha256(dp)
        if s_hash != d_hash:
            hashdiff.append(f"{rel} src={s_hash} dest={d_hash}")
            log(f"HASH-DIFF {rel}")
            continue
        verified += 1
        if i % 20 == 0 or i == len(src_files):
            log(f"verified {i}/{len(src_files)} ({total_bytes / 1e9:.1f} GB read)")

    elapsed = time.time() - t0
    log(f"verified={verified} missing={len(missing)} sizediff={len(sizediff)} hashdiff={len(hashdiff)}")
    log(f"elapsed={elapsed:.1f}s  bytes_compared={total_bytes / 1e9:.2f} GB")

    ok = not missing and not sizediff and not hashdiff and verified == len(src_files)
    log(f"VERIFY RESULT ok={ok}")

    if not ok:
        log("NOT deleting src: verification failed")
        return 1

    if not args.delete:
        log("verification clean; re-run with --delete to remove the C: source")
        return 0

    # free space check on C: before report
    before = shutil.disk_usage("C:\\").free
    log("DELETING src tree ...")
    try:
        shutil.rmtree(SRC)
        log("src tree deleted")
    except Exception as exc:  # noqa: BLE001
        log(f"delete failed: {exc}")
        return 1
    after = shutil.disk_usage("C:\\").free
    log(f"C: free before={before / 1e9:.2f}GB after={after / 1e9:.2f}GB delta={(after - before) / 1e9:.2f}GB")
    log(f"src still exists={SRC.exists()}")

    # prune the now-empty parent if it holds nothing else
    parent = SRC.parent
    try:
        leftovers = list(parent.iterdir())
    except Exception:
        leftovers = []
    log(f"hub dir siblings={[p.name for p in leftovers]}")
    log("FINALIZE END")
    return 0


if __name__ == "__main__":
    sys.exit(main())
