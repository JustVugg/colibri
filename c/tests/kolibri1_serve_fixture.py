#!/usr/bin/env python3
"""A SERVE-ready copy of the tiny kolibri1 fixture: the fixture plus a tokenizer.json.

The tiny model has vocab 320, so its tokenizer must never produce an id >= 320. This builds
one from tests/tok_kolibri1_tiny.json (the release's real pre-tokenizer, decoder and the
byte-level alphabet) keeping only the first 30 merges and putting the ChatML / think / tool
added tokens at ids 290..298. `tokenizer_config`-free; the engine reads tokenizer.json only.

make_serve_fixture(dst, eos=...) generates the fixture (release layout), writes the tokenizer
and rewrites config.json: eos_token_id = <|im_end|> (292), max_position_embeddings 4096 (the
gateway prompts with a tools block are longer than 1024 byte-level tokens). The oracle
fixture is NOT changed (it keeps eos 1 and 1024); this copy is for SERVE tests only.
"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TOOL = ROOT / "tools" / "make_kolibri1_tiny.py"
SRC_TOKENIZER = HERE / "tok_kolibri1_tiny.json"

ADDED = {
    "<|endoftext|>": (290, True), "<|im_start|>": (291, True), "<|im_end|>": (292, True),
    "<think>": (293, False), "</think>": (294, False),
    "<tool_call>": (295, False), "</tool_call>": (296, False),
    "<tool_response>": (297, False), "</tool_response>": (298, False),
}
IM_END = 292
MERGES = 30


def build_tokenizer():
    t = json.loads(SRC_TOKENIZER.read_text(encoding="utf-8"))
    vocab = {tok: i for tok, i in t["model"]["vocab"].items() if i < 256}
    assert len(vocab) == 256
    merges = []
    for a, b in t["model"]["merges"]:
        if a in vocab and b in vocab and len(a) == 1 and len(b) == 1 and len(merges) < MERGES:
            merges.append([a, b])
            vocab[a + b] = 256 + len(merges) - 1
    assert len(merges) == MERGES
    t["model"]["vocab"] = vocab
    t["model"]["merges"] = merges
    t["added_tokens"] = [
        {"id": i, "content": c, "single_word": False, "lstrip": False, "rstrip": False,
         "normalized": False, "special": special}
        for c, (i, special) in sorted(ADDED.items(), key=lambda kv: kv[1][0])]
    return t


VENDOR_GENERATION_CONFIG = {"temperature": 1.0, "top_p": 0.97, "top_k": 128, "do_sample": True,
                             "eos_token_id": [127906, 127901], "pad_token_id": 127901}


def make_serve_fixture(dst, topk=None, gen_config=None):
    dst = Path(dst)
    cmd = [sys.executable, str(TOOL), "--out", str(dst), "--layout", "release", "--force"]
    if topk:
        cmd += ["--topk", str(topk)]
    subprocess.run(cmd, check=True, capture_output=True)
    (dst / "tokenizer.json").write_text(json.dumps(build_tokenizer(), ensure_ascii=False), encoding="utf-8")
    cfg = json.loads((dst / "config.json").read_text())
    cfg["eos_token_id"] = IM_END
    cfg["max_position_embeddings"] = 4096
    (dst / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    if gen_config is not False:   # the release ships one; values below are the vendor's
        (dst / "generation_config.json").write_text(json.dumps(
            gen_config or VENDOR_GENERATION_CONFIG, indent=2) + "\n")
    return dst


def add_tokenizer_only(dst):
    """tokenizer.json into an existing fixture copy, config untouched (the oracle copies)"""
    dst = Path(dst)
    (dst / "tokenizer.json").write_text(json.dumps(build_tokenizer(), ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    make_serve_fixture(sys.argv[1])
