#!/usr/bin/env python3
"""Un'immagine attraversa il protocollo e cambia la risposta.

Fino a ieri la torre vision c'era e non era raggiungibile: la CLI voleva patch
gia' estratte, il protocollo portava solo testo e il gateway buttava via le
immagini in silenzio, che e' il modo peggiore di non supportarle -- l'utente
manda una foto, riceve una risposta, e la risposta parla del nulla.

Questo test segue la catena intera: si preprocessa un'immagine con lo stesso
codice che usa il gateway, la si manda col frame IMAGE, e si chiede al motore
di rispondere a un prompt che contiene i segnaposto.

La verifica non e' che risponda: risponderebbe comunque, anche ignorando i
pixel. E' che **due immagini diverse diano due risposte diverse**. Se il frame
arrivasse e venisse scartato, o gli embedding finissero nelle posizioni
sbagliate, il modello continuerebbe a parlare e le due risposte sarebbero
identiche: e' l'unica differenza che distingue una vision collegata da una
vision finta.

USO:
  python3 tests/test_glm53_vision_serve.py --binary ./glm53 --fixture ~/glm53_mm_tiny
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

IMAGE_OPEN, IMAGE_TOKEN, IMAGE_CLOSE = "<|begin_of_image|>", "<|image|>", "<|end_of_image|>"


def make_image(seed, side, numpy):
    """Un'immagine deterministica, senza file su disco."""
    generator = numpy.random.default_rng(seed)
    return generator.integers(0, 256, (side, side, 3), dtype=numpy.uint8)


def read_line(stream):
    line = stream.readline()
    if not line:
        raise AssertionError("il motore ha chiuso lo stream prima del previsto")
    return line.decode("utf-8", "replace").rstrip("\n")


def ask(process, request_id, prompt, image):
    """IMAGE piu' SUBMIT, e la risposta in byte."""
    patches, grid_h, grid_w = image
    blob = patches.tobytes()
    process.stdin.write(
        f"IMAGE {request_id} {len(blob)} {grid_h} {grid_w}\n".encode() + blob + b"\n")
    body = prompt.encode("utf-8")
    process.stdin.write(
        f"SUBMIT {request_id} 0 {len(body)} 4 0.0 1.0\n".encode() + body + b"\n")
    process.stdin.flush()

    pieces = []
    while True:
        line = read_line(process.stdout)
        if line.startswith("DATA "):
            count = int(line.split()[2])
            pieces.append(process.stdout.read(count + 1)[:count])
        elif line.startswith("ERROR "):
            raise AssertionError(f"il motore ha rifiutato: {line}")
        elif line.startswith("DONE "):
            return b"".join(pieces)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    arguments = parser.parse_args()
    if not (arguments.fixture / "ref.json").exists():
        # Un traceback su un file che manca fa sembrare rotto il
        # motore; chi arriva per la prima volta non puo' distinguere
        # le due cose. Il generatore vuole transformers 5.16.1.
        print(f"SKIP: manca {arguments.fixture}; generalo con\n"
              f"  python3 tools/make_glm53_multimodal_tiny.py --output <dir>")
        return 0

    try:
        import numpy
        from PIL import Image
    except ImportError as problem:
        print(f"SKIP: servono numpy e Pillow ({problem}); niente e' stato verificato")
        return 0
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    from glm53_image import preprocess, load_config

    settings = load_config(arguments.fixture)
    merge = settings.get("merge_size", 2)
    patch = settings.get("patch_size", 14)
    # La misura piu' piccola che soddisfa il minimo di 16 token immagine.
    side = patch * merge * 4

    images = [preprocess(Image.fromarray(make_image(seed, side, numpy)), arguments.fixture)
              for seed in (1, 2)]
    tokens = (images[0][1] // merge) * (images[0][2] // merge)
    if images[1][1:] != images[0][1:]:
        print("SKIP: le due immagini di prova hanno griglie diverse")
        return 0

    prompt = "gu" + IMAGE_OPEN + IMAGE_TOKEN * tokens + IMAGE_CLOSE + "xy"
    environment = {**os.environ, "SERVE": "1", "SERVE_BATCH": "1",
                   "SNAP": str(arguments.fixture), "GLM53_BITS": "32"}
    process = subprocess.Popen([os.path.abspath(arguments.binary)],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.DEVNULL, env=environment)
    try:
        if "READY" not in read_line(process.stdout):
            print("FAIL: nessun READY")
            return 1
        read_line(process.stdout)                       # STAT
        first = ask(process, 1, prompt, images[0])
        second = ask(process, 2, prompt, images[1])
        process.stdin.close()
        process.wait(timeout=120)
    finally:
        if process.poll() is None:
            process.kill()

    if not first:
        print("FAIL: nessun token generato con un'immagine")
        return 1
    if first == second:
        print(f"FAIL: due immagini diverse danno la stessa risposta ({first!r});\n"
              f"  i pixel non stanno arrivando al modello")
        return 1

    print(f"PASS GLM-5.3 vision sul protocollo: {tokens} token immagine da una "
          f"{side}x{side}, due immagini diverse danno risposte diverse "
          f"({first!r} contro {second!r})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
