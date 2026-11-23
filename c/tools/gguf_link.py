#!/usr/bin/env python3
"""A catalog GGUF the setup downloaded, imported where it lies: `gguf_link.py MODEL_DIR`.

The setup's post-download step for the catalog's GGUF entries (setup_catalog.py, post_install):
the model's GGUF in MODEL_DIR (or a folder of it, split models from their first part) through
gguf_import.py in link mode, with MODEL_DIR as the output, so config.json, the tokenizer and
gguf.json sit next to the files they point at. The importer finds the vision tower (mmproj-*.gguf)
and an MTP head (mtp-*.gguf) next to the model on its own.
"""
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def model_gguf(folder):
    """The model's GGUF under `folder`: not a vision tower, an MTP or draft head, or a later part."""
    found = []
    for path in sorted(folder.rglob("*.gguf")):
        name = path.name.lower()
        if name.startswith(("mmproj", "mtp-", "dspark-")) or "imatrix" in name:
            continue
        part = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", name)
        if part and part.group(1) != "00001":
            continue
        found.append(path)
    if len(found) != 1:
        sys.exit(f"{folder}: {'no model GGUF' if not found else 'more than one model GGUF: ' + ', '.join(p.name for p in found)}")
    return found[0]


def main(argv):
    if len(argv) != 2:
        sys.exit("usage: gguf_link.py MODEL_DIR")
    folder = Path(argv[1]).resolve()
    gguf = model_gguf(folder)
    return subprocess.call([sys.executable, str(HERE / "gguf_import.py"), str(gguf), str(folder), "--mode", "link"])


if __name__ == "__main__":
    sys.exit(main(sys.argv))
