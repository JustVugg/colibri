#!/usr/bin/env python3
"""Lo streaming degli esperti legge davvero i byte giusti.

Sul modello vero gli esperti sono 171 GB e non entrano in RAM: il motore tiene
una tabella dei loro indirizzi, una cache LRU per layer, e costruisce le
matrici puntando dentro allo slot appena letto. Un offset sbagliato, una forma
scambiata fra gate e down, una vittima scelta male: nessuna di queste cose
fa cadere il programma. Calcola su altri byte e risponde lo stesso.

Il confronto e' fra due fixture con gli stessi identici numeri, uno col
contenitore int4 che il motore legge in streaming e uno con quegli stessi
valori gia' dequantizzati, che il motore carica residenti. La quantizzazione e'
avvenuta in entrambi, quindi l'unica differenza rimasta e' da dove arrivano i
byte, e le risposte devono coincidere.

Si esegue due volte sul contenitore int4: con un budget largo, dove ogni
esperto si legge una volta sola, e con un budget da uno slot, dove la cache
sbatte e gli stessi esperti vengono riletti in continuazione. Se il secondo
giro rispondesse diverso, sarebbe l'eviction a sbagliare.

Il conteggio dei miss viene controllato perche' sia diverso da zero: un motore
che avesse silenziosamente caricato tutto in RAM passerebbe ogni altro
controllo di questo test.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def run(binary, fixture, patches, grid, budget=None):
    environment = {**os.environ, "GLM53_BITS": "32"}
    if budget is not None:
        environment["GLM53_EXPERT_GB"] = budget
    command = [binary, "--model", str(fixture), "--ids", patches["ids"]]
    if patches["file"]:
        command += ["--patches", patches["file"], "--grid", grid]
    command += ["--greedy", "3"]
    result = subprocess.run(command, capture_output=True, text=True, check=True,
                            env=environment)
    parsed = {}
    for line in result.stdout.splitlines():
        if line.strip():
            parsed[line.split()[0]] = line.split()[1:]
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--quantized", type=Path, required=True,
                        help="fixture con gli esperti nel contenitore int4")
    parser.add_argument("--dequantized", type=Path, required=True,
                        help="fixture con quegli stessi valori in f32")
    arguments = parser.parse_args()
    if not (arguments.quantized / "ref.json").exists():
        # Un traceback su un file che manca fa sembrare rotto il
        # motore; chi arriva per la prima volta non puo' distinguere
        # le due cose. Il generatore vuole transformers 5.16.1.
        print(f"SKIP: manca {arguments.quantized}; generalo con\n"
              f"  python3 tools/make_glm53_streaming_pair.py --fixture <mm> --output <dir>")
        return 0

    reference = json.loads((arguments.quantized / "ref.json").read_text())
    grid_h, grid_w = reference.get("grid", (0, 0))
    image = arguments.quantized / "patches.f32"
    patches = {"ids": ",".join(str(t) for t in reference["prompt"]),
               "file": str(image) if image.exists() else None}
    grid = f"{grid_h}x{grid_w}"

    resident = run(arguments.binary, arguments.dequantized, patches, grid)
    wide = run(arguments.binary, arguments.quantized, patches, grid, budget="8")
    narrow = run(arguments.binary, arguments.quantized, patches, grid, budget="0.000001")

    if "experts" not in wide or "experts" not in narrow:
        print("FAIL: il motore non ha usato lo streaming sul contenitore int4")
        return 1
    misses = {"largo": int(wide["experts"][3]), "stretto": int(narrow["experts"][3])}
    if misses["largo"] < 1:
        print("FAIL: nessuna lettura da disco, gli esperti non sono stati streammati")
        return 1
    if misses["stretto"] <= misses["largo"]:
        print(f"FAIL: con un solo slot i miss non aumentano "
              f"({misses['stretto']} contro {misses['largo']}): la cache non sfratta")
        return 1

    for label, run_output in (("budget largo", wide), ("budget stretto", narrow)):
        for field in ("teacher_forcing", "greedy"):
            if run_output[field] != resident[field]:
                print(f"FAIL {field} con {label}\n"
                      f"  streaming: {run_output[field]}\n"
                      f"  residente: {resident[field]}")
                return 1

    print(f"PASS GLM-5.3 streaming: stessi token dei pesi residenti, "
          f"{misses['largo']} letture col budget largo e {misses['stretto']} "
          f"con un solo slot per layer")
    return 0


if __name__ == "__main__":
    sys.exit(main())
