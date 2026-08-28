#!/usr/bin/env python3
"""Il percorso Vulkan di GLM-5.3 risponde come la CPU.

Con COLI_VULKAN=1 le matrici residenti passano dal backend Vulkan invece che
dai kernel di quant.h. Sono due implementazioni diverse della stessa moltiplicazione,
e la sola cosa che conta e' che diano gli stessi token: un kernel che sbaglia
la decodifica dei nibble, il verso dei gruppi o l'ordine delle scale non da'
errore, da' un modello che risponde peggio.

Il confronto e' sui token, non sui logit. Un backend puo' sommare in un altro
ordine e finire su cifre diverse in fondo al float senza che questo cambi
niente; se cambia il token, invece, ha cambiato la risposta.

Gli esperti restano sulla CPU di proposito: arrivano dal disco a ogni uso,
quindi caricarli sul device costerebbe quanto leggerli. Quelli vogliono un
livello residente in VRAM, che e' un'altra cosa e non e' questo test.

Il test si dichiara SALTATO quando il binario non e' stato costruito con VK=1 o
non c'e' nessun device: non ha verificato niente e dirlo verde sarebbe peggio.

USO:
  make VK=1 glm53
  python3 tests/test_glm53_vulkan.py --binary ./glm53 --fixture ~/glm53_mm_tiny
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path


def run(binary, fixture, reference, vulkan, shaders):
    environment = {**os.environ, "GLM53_BITS": "4"}
    if vulkan:
        environment["COLI_VULKAN"] = "1"
        environment["COLI_VK_SHADERS"] = str(shaders)
    else:
        environment.pop("COLI_VULKAN", None)
    command = [binary, "--model", str(fixture),
               "--ids", ",".join(str(t) for t in reference["prompt"]),
               "--greedy", "4"]
    image = fixture / "patches.f32"
    if image.exists():
        grid_h, grid_w = reference["grid"]
        command += ["--patches", str(image), "--grid", f"{grid_h}x{grid_w}"]
    result = subprocess.run(command, capture_output=True, text=True, check=True,
                            env=environment)
    lines = {line.split()[0]: line.split()[1:]
             for line in result.stdout.splitlines() if line.strip()}
    return lines, result.stderr


def main() -> int:
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--shaders", type=Path,
                        default=Path(__file__).resolve().parents[1] / "shaders")
    arguments = parser.parse_args()
    if not (arguments.fixture / "ref.json").exists():
        # Un traceback su un file che manca fa sembrare rotto il
        # motore; chi arriva per la prima volta non puo' distinguere
        # le due cose. Il generatore vuole transformers 5.16.1.
        print(f"SKIP: manca {arguments.fixture}; generalo con\n"
              f"  python3 tools/make_glm53_multimodal_tiny.py --output <dir>")
        return 0
    binary = os.path.abspath(arguments.binary)
    reference = json.loads((arguments.fixture / "ref.json").read_text())

    cpu, _ = run(binary, arguments.fixture, reference, False, arguments.shaders)
    gpu, notes = run(binary, arguments.fixture, reference, True, arguments.shaders)

    if "attivo" not in notes:
        reason = ("il binario non e' costruito con VK=1"
                  if "Vulkan:" not in notes else "nessun device Vulkan utilizzabile")
        print(f"SKIP: {reason}; il percorso Vulkan non e' stato verificato")
        return 0

    device = next((line for line in notes.splitlines() if "[VK] ready:" in line), "")
    for field in ("teacher_forcing", "greedy"):
        if cpu.get(field) != gpu.get(field):
            print(f"FAIL {field}\n  CPU:    {cpu.get(field)}\n  Vulkan: {gpu.get(field)}")
            return 1

    print(f"PASS GLM-5.3 Vulkan: stessi token della CPU su "
          f"{len(cpu['teacher_forcing'])} posizioni e {len(cpu['greedy'])} passi greedy"
          f"{' — ' + device.split('ready:')[1].split(',')[0].strip() if device else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
