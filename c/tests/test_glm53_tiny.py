#!/usr/bin/env python3
"""Il motore GLM-5.3 contro l'oracolo tiny di solo testo.

Il fixture lo produce tools/make_glm53_tiny.py, che e' un secondo parere: e'
scritto da un'altra mano a partire dallo stesso Glm5NextTextModel, e vale
esattamente perche' non l'abbiamo scritto noi. Copre KDA, DSA, mHC, l'FFN
denso, il MoE routed, l'indexer, l'attenzione sparsa e i dati del contratto
FP8.

Confronta cio' che un utente vede davvero -- i token -- e non solo i numeri
interni: teacher forcing su ogni posizione, generazione greedy, e i logit
dell'ultima posizione entro una tolleranza stretta. Un motore puo' avere logit
quasi giusti e scegliere comunque il token sbagliato, quindi entrambi contano.

Le due domande sono pero' diverse e vanno tenute separate. "Il motore
implementa il modello" si prova in f32, dove i logit devono coincidere fino
all'ultima cifra utile. "La quantizzazione conserva le risposte" si prova a bit
ridotti, dove i logit sono legittimamente diversi e a dover restare identici
sono i token. Chiedere gli stessi logit a int4 vorrebbe dire chiedere che la
quantizzazione non quantizzi.

Il confronto coi token dell'oracolo si fa in f32, e il test impone lui la
precisione al processo figlio invece di ereditarla dall'ambiente: cosi' il
risultato non dipende da come e' impostata la shell di chi lo lancia. Con
--bits si puo' rieseguire a precisione ridotta, ma quella e' una misura, non un
cancello: questi modelli hanno hidden 128, cioe' due soli gruppi per riga, e
pesi casuali senza struttura. Quanto int4 conservi le risposte si misura sul
modello vero, dove le righe sono lunghe 4096 e i pesi vogliono dire qualcosa.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# Quanto puo' spostare i logit ciascuna precisione dei pesi densi. Sono soglie
# misurate su questo fixture, non tolleranze di comodo: servono ad accorgersi
# se un giorno la quantizzazione peggiora, non a far passare il test.
LOGIT_TOLERANCE = {32: 2e-5, 8: 3e-2, 4: 5e-1}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--logit-tolerance", type=float, default=None)
    parser.add_argument("--bits", type=int, default=32,
                        choices=(4, 8, 32),
                        help="precisione dei densi; 32 e' il confronto "
                             "con l'oracolo, il resto e' una misura")
    arguments = parser.parse_args()

    bits = arguments.bits
    if arguments.logit_tolerance is None:
        arguments.logit_tolerance = LOGIT_TOLERANCE.get(bits, 5e-1)

    if not (arguments.fixture / "ref.json").exists():
        # Un traceback su un file che manca fa sembrare rotto il motore.
        print(f"SKIP: manca {arguments.fixture}/ref.json; generalo con\n"
              f"  pip install -r tools/requirements-glm53-tiny.txt\n"
              f"  python3 tools/make_glm53_tiny.py --output {arguments.fixture}")
        return 0
    reference = json.loads((arguments.fixture / "ref.json").read_text())
    prompt = ",".join(str(token) for token in reference["prompt_ids"])
    expected_forcing = reference["teacher_forcing_ids"]
    expected_greedy = reference["greedy_new_ids"]

    result = subprocess.run(
        [arguments.binary, "--model", str(arguments.fixture), "--ids", prompt,
         "--greedy", str(len(expected_greedy)), "--logits"],
        capture_output=True, text=True, check=True,
        env={**os.environ, "GLM53_BITS": str(bits)})

    lines = {line.split()[0]: line.split()[1:] for line in result.stdout.splitlines() if line.strip()}
    forcing = [int(value) for value in lines["teacher_forcing"]]
    greedy = [int(value) for value in lines["greedy"]]
    logits = [float(value) for value in lines["last_logits"]]

    if forcing != expected_forcing:
        print(f"FAIL teacher forcing\n  ottenuto: {forcing}\n  atteso:   {expected_forcing}")
        return 1
    if greedy != expected_greedy:
        print(f"FAIL greedy\n  ottenuto: {greedy}\n  atteso:   {expected_greedy}")
        return 1
    worst = max(abs(a - b) for a, b in zip(logits, reference["last_logits"], strict=True))
    if worst > arguments.logit_tolerance:
        print(f"FAIL logit: max abs {worst:.3g} oltre {arguments.logit_tolerance:.3g}")
        return 1

    print(f"PASS GLM-5.3 tiny: {len(forcing)} posizioni teacher-forced esatte, "
          f"{len(greedy)} token greedy esatti, densi a {bits} bit, "
          f"logit entro {worst:.3g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
