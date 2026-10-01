#!/usr/bin/env python3
"""Uno scatto (pin) non si rimette sopra righe che un altro ramo ha riscritto.

Lo scatto di GLM-5.3 salva lo stato ricorrente dei layer KDA, non le righe dei
layer DSA: quelle restano dove sono, indicizzate per posizione. slot_pin_restore
controlla che la sessione sia la stessa e che lo scatto non sia piu' profondo
di `filled`, ma non che le righe fino a quella profondita' siano ancora le sue.

La sequenza che lo rompe resta tutta nella stessa sessione:

  1. P      pin=1   scatto a |P|
  2. P+A    pin=1   estende, scatto a |P+A|
  3. P+B            diverge da A: si rimette lo scatto P e si macina B sopra
                    le righe |P|.. -- la sessione e' la stessa, gli scatti restano
  4. P+A+X          lo scatto P+A combacia con la richiesta ed e' <= filled,
                    quindi si rimette: stato KDA di P+A, righe DSA di P+B

La risposta resta plausibile, quindi la prova e' un confronto a coppie: i
punteggi (ECHO) della coda X devono essere identici a quelli di un motore
freddo sullo stesso prompt. Il controllo senza il passo 3 li da' identici; col
passo 3 no.

Come per gli altri harness glm53, la fixture si genera con
  python3 tools/make_glm53_multimodal_tiny.py --output <dir>
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

PROMPT, BRANCH_A, BRANCH_B, TAIL = (
    "the quick brown", " fox jumps", " cat sleeps all day", " over it")


class Engine:
    def __init__(self, binary, fixture):
        env = {**os.environ, "SERVE": "1", "SERVE_BATCH": "1", "SNAP": str(fixture),
               "GLM53_BITS": "32"}
        self.p = subprocess.Popen([binary], env=env, stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        while "READY" not in self.line():
            pass

    def line(self):
        line = self.p.stdout.readline()
        if not line:
            raise AssertionError("il motore ha chiuso lo stream prima del previsto")
        return line.decode("utf-8", "replace").rstrip("\n")

    def score(self, request_id, text, pin=False):
        """Legge il prompt senza generare (max_tokens 0, logprobs=1): i punteggi
        ECHO per posizione, e la riga che chiude la richiesta."""
        body = text.encode()
        extension = " logprobs=1" + (" pin=1" if pin else "")
        self.p.stdin.write(f"SUBMIT {request_id} 0 {len(body)} 0 0 1{extension}\n".encode()
                           + body + b"\n")
        self.p.stdin.flush()
        echoes = {}
        while True:
            fields = self.line().split()
            if fields and fields[0] in ("ECHO", "DATA"):
                self.p.stdout.read(int(fields[2]) + 1)
                if fields[0] == "ECHO" and fields[4] not in ("nan", "-nan"):
                    echoes[int(fields[3])] = fields[4]
            if fields and fields[0] in ("DONE", "ERROR"):
                return " ".join(fields), echoes

    def close(self):
        self.p.stdin.close()
        self.p.wait(timeout=60)


def tail_scores(binary, fixture, branch_in_between):
    warm = Engine(binary, fixture)
    try:
        warm.score(1, PROMPT, pin=True)
        warm.score(2, PROMPT + BRANCH_A, pin=True)
        if branch_in_between:
            warm.score(3, PROMPT + BRANCH_B)
        done, scores = warm.score(4, PROMPT + BRANCH_A + TAIL)
    finally:
        warm.close()
    if not done.startswith("DONE 4 "):
        raise AssertionError(f"richiesta 4 chiusa con {done!r}")
    return scores


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    arguments = parser.parse_args()
    if not (arguments.fixture / "config.json").exists():
        print(f"SKIP: manca {arguments.fixture}; generala con\n"
              f"  python3 tools/make_glm53_multimodal_tiny.py --output <dir>")
        return 2
    binary = os.path.abspath(arguments.binary)

    cold = Engine(binary, arguments.fixture)
    try:
        _, fresh = cold.score(1, PROMPT + BRANCH_A + TAIL)
    finally:
        cold.close()
    tail = sorted(p for p in fresh if p >= len((PROMPT + BRANCH_A).encode()))
    if not tail:
        print("FAIL: il motore freddo non ha dato punteggi per la coda")
        return 1

    for branch_in_between in (False, True):
        scores = tail_scores(binary, arguments.fixture, branch_in_between)
        wrong = [p for p in tail if scores.get(p) != fresh[p]]
        if wrong:
            where = "con un ramo P+B in mezzo" if branch_in_between else "senza rami"
            print(f"FAIL: {where}, {len(wrong)} posizioni su {len(tail)} della coda "
                  f"non combaciano con un motore freddo (prima: {wrong[0]}, "
                  f"{scores.get(wrong[0])} contro {fresh[wrong[0]]}) -- uno scatto "
                  f"e' stato rimesso sopra righe che non sono piu' le sue")
            return 1

    print(f"PASS GLM-5.3 pin: le {len(tail)} posizioni della coda combaciano con un "
          f"motore freddo, anche dopo un ramo che ha riscritto le righe dello scatto")
    return 0


if __name__ == "__main__":
    sys.exit(main())
