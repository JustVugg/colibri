#!/usr/bin/env python3
"""Il percorso completo delle immagini in Qwen3.8: frame IMAGE, torre, generazione.

Non verifica che il modello RISPONDA: con pesi casuali risponderebbe comunque, e
risponderebbe identico anche ignorando del tutto l'immagine. Verifica che **due
immagini diverse diano due risposte diverse**, che e' l'unica domanda a cui una
fixture casuale sa rispondere onestamente, ed e' anche quella che smaschera il
difetto piu' probabile: patch caricate, torre eseguita, e poi il risultato
buttato via da qualche parte fra il merger e gli embedding.

Poi controlla i rifiuti, perche' accettare un'immagine sbagliata e' peggio che
rifiutarla: un conteggio di byte che non corrisponde alla griglia, e un prompt in
cui i segnaposto non sono quanti la griglia ne produce.

USO:
  python3 tests/test_qwen38_vision_serve.py --binary ./qwen38 --fixture ./qwen38_mm_tiny
"""
import argparse
import json
import struct
import subprocess
import sys
from pathlib import Path


def read_until(process, prefixes, limit=400):
    """Legge righe finche' non ne arriva una che comincia per uno dei prefissi."""
    for _ in range(limit):
        line = process.stdout.readline()
        if not line:
            return None
        # Le righe di controllo del mux sono racchiuse da \x01: senza toglierli
        # un confronto per prefisso non trova mai nulla e il test aspetta per sempre.
        text = line.decode("utf-8", "replace").strip("\r\n\x01")
        for prefix in prefixes:
            if text.startswith(prefix):
                return text
    return None


def turn(process, request_id, prompt_ids, patches, grid_h, grid_w, image_token):
    """Manda IMAGE + SUBMIT e restituisce (verdetto, testo raccolto)."""
    payload = bytes(bytearray([min(t, 255) for t in prompt_ids]))
    if patches is not None:
        blob = struct.pack(f"<{len(patches)}f", *patches)
        process.stdin.write(
            f"IMAGE {request_id} {len(blob)} {grid_h} {grid_w}\n".encode() + blob + b"\n")
    header = f"SUBMIT {request_id} 0 {len(payload)} 4 0.0 1.0\n".encode()
    process.stdin.write(header + payload + b"\n")
    process.stdin.flush()
    first = read_until(process, ("ACCEPT", "ERROR"))
    if first is None or first.startswith("ERROR"):
        return (first or "nessuna risposta"), ""
    collected = []
    for _ in range(200):
        line = process.stdout.readline()
        if not line:
            break
        text = line.decode("utf-8", "replace").strip("\r\n\x01")
        if text.startswith("DATA"):
            collected.append(text)
        if text.startswith("DONE") or text.startswith("ERROR"):
            return text, "\n".join(collected)
    return "senza DONE", "\n".join(collected)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    arguments = parser.parse_args()

    meta_path = arguments.fixture / "vision_meta.json"
    if not arguments.binary.exists() or not meta_path.exists():
        print(f"SKIP: serve {arguments.binary} e la fixture; generala con "
              f"tools/make_qwen38_multimodal_tiny.py")
        return 0
    meta = json.loads(meta_path.read_text())
    grid_h, grid_w = meta["grid_h"], meta["grid_w"]
    features, image_token = meta["features"], meta["image_token"]
    tokens = meta["tokens"]
    count = grid_h * grid_w * features

    environment = {"SNAP": str(arguments.fixture), "SERVE": "1", "SERVE_BATCH": "1",
                   "OMP_NUM_THREADS": "2", "PATH": "/usr/bin:/bin"}
    process = subprocess.Popen([str(arguments.binary.resolve()), "1", "4"],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.DEVNULL, env=environment)
    failures = 0
    try:
        if read_until(process, ("READY",)) is None:
            print("FAIL il motore non ha mai detto READY")
            return 1
        print("ok   il motore e' pronto")

        prompt = [image_token] * tokens + [3, 4]

        # Due immagini deliberatamente diverse: una a gradiente crescente, una a
        # gradiente opposto. Se il risultato della torre non arrivasse agli
        # embedding, le due risposte sarebbero identiche.
        first = [((i % 97) / 97.0) for i in range(count)]
        second = [1.0 - value for value in first]

        verdict_a, text_a = turn(process, "a1", prompt, first, grid_h, grid_w, image_token)
        verdict_b, text_b = turn(process, "b1", prompt, second, grid_h, grid_w, image_token)
        for label, verdict in (("prima immagine", verdict_a), ("seconda immagine", verdict_b)):
            if not verdict.startswith("DONE"):
                print(f"FAIL {label}: {verdict}")
                failures += 1
        if not failures:
            if text_a and text_b and text_a != text_b:
                print("ok   due immagini diverse danno due risposte diverse")
            else:
                print("FAIL le due immagini hanno dato la stessa risposta: la torre "
                      "gira ma il suo risultato non arriva agli embedding")
                failures += 1

        # Byte che non corrispondono alla griglia: va rifiutato, non arrotondato.
        verdict, _ = turn(process, "c1", prompt, first[:-features], grid_h, grid_w, image_token)
        if verdict.startswith("ERROR"):
            print("ok   un conteggio di byte incoerente con la griglia e' rifiutato")
        else:
            print(f"FAIL byte incoerenti accettati: {verdict}")
            failures += 1

        # Segnaposto in numero sbagliato: il prompt e la griglia devono concordare.
        verdict, _ = turn(process, "d1", [image_token] * (tokens - 1) + [3],
                          first, grid_h, grid_w, image_token)
        if verdict.startswith("ERROR"):
            print("ok   un prompt con troppi pochi segnaposto e' rifiutato")
        else:
            print(f"FAIL segnaposto mancanti accettati: {verdict}")
            failures += 1

        # Senza immagine il motore deve continuare a funzionare come prima.
        verdict, _ = turn(process, "e1", [3, 4, 5], None, 0, 0, image_token)
        if verdict.startswith("DONE"):
            print("ok   una richiesta di solo testo funziona ancora")
        else:
            print(f"FAIL il solo testo si e' rotto: {verdict}")
            failures += 1
    finally:
        try:
            process.stdin.close()
        except Exception:
            pass
        process.terminate()
        process.wait(timeout=10)

    print()
    if failures:
        print(f"TEST FAIL ({failures} casi)")
        return 1
    print("percorso immagini Qwen3.8: frame, torre e generazione collegati")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
