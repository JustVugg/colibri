"""The tiny MiMo fixture, served: the same weights plus a tokenizer.

tools/make_mimo_tiny.py ships no tokenizer.json on purpose -- the vendor oracle
(tests/mimo_tiny_harness.py) reads raw token ids on stdout and refuses a fixture
that has one. Serve mode goes through text, so the serve tests work on a copy:
the fixture's files plus a one-token-per-byte tokenizer, with the three vision
markers as added tokens at the ids the fixture's config gives them, so an image
prompt can be written as text.

The copy is made once per process, in a temporary directory that is removed at
exit, and the source fixture is never touched.
"""
import atexit
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prefix_serve_harness import byte_level_vocab  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
ENGINE = Path(os.environ.get("MIMO_BINARY",
                             HERE / ("mimo.exe" if sys.platform == "win32" else "mimo")))
FIXTURE = Path(os.environ.get("MIMO_TINY", HERE / "mimo_tiny"))

IMAGE_PAD = "<|image_pad|>"
VISION_START = "<|vision_start|>"
VISION_END = "<|vision_end|>"

_served = None


def available():
    return ENGINE.is_file() and (FIXTURE / "config.json").is_file()


def has_vision():
    return (FIXTURE / "patches.f32").is_file() and (FIXTURE / "image.json").is_file()


def served_fixture():
    """The serve-mode copy of FIXTURE, built on first use."""
    global _served
    if _served is not None:
        return _served
    root = Path(tempfile.mkdtemp(prefix="mimo-serve-"))
    atexit.register(shutil.rmtree, root, True)
    out = root / "fixture"
    shutil.copytree(FIXTURE, out)
    config = json.loads((FIXTURE / "config.json").read_text())
    added = []
    for key, text in (("image_token_id", IMAGE_PAD), ("vision_start_token_id", VISION_START),
                      ("vision_end_token_id", VISION_END)):
        if isinstance(config.get(key), int):
            added.append({"id": config[key], "content": text, "special": True})
    (out / "tokenizer.json").write_text(json.dumps({
        "version": "1.0",
        "added_tokens": added,
        "model": {"type": "BPE", "vocab": byte_level_vocab(256), "merges": []},
    }), encoding="utf-8")
    _served = out
    return out


def image_payload():
    """(patch bytes, grid_h, grid_w, pad count) of the fixture's picture."""
    meta = json.loads((FIXTURE / "image.json").read_text())
    return ((FIXTURE / "patches.f32").read_bytes(), meta["grid_h"], meta["grid_w"],
            meta["tokens"])


def engine_env(**extra):
    """What every serve test runs the engine with: the served copy, a small
    context, two threads and no OpenMP re-exec."""
    env = {"SNAP": str(served_fixture()), "SERVE": "1", "CTX": "256",
           "OMP_NUM_THREADS": "2", "COLI_NO_OMP_TUNE": "1"}
    env.update(extra)
    return env
