#!/usr/bin/env python3
"""Il renderer di Qwen3.6 nel gateway, contro chat_template.jinja ufficiale.

Il gateway rende i prompt a mano invece di far girare jinja a ogni richiesta, e
quella scelta si paga in un modo solo: la copia scritta a mano puo' scostarsi
dall'originale senza che nessuno se ne accorga, perche' il modello risponde
comunque. Qui il template vero viene reso con jinja2 e confrontato byte per byte
con quello che produce il gateway, senza strumenti (che il motore qwen36 non
espone) e sui due rami del blocco di ragionamento, piu' il turno aperto della
prosecuzione.

Se manca il template o jinja2, il test si dichiara SALTATO invece di passare: un
test che non ha trovato il suo riferimento non ha verificato niente, e dirlo
verde sarebbe peggio che non averlo -- percio' un salto esce con codice 2,
distinto dallo 0 di un confronto riuscito.

RIFERIMENTO (scaricato 2026-09-10):
  repo     Qwen/Qwen3.6-35B-A3B      (repo vendor; il container colibri dava 404)
  file     chat_template.jinja
  sha256   e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259
  hf download Qwen/Qwen3.6-35B-A3B chat_template.jinja

USO:
  python3 tests/test_qwen36_chat_template.py --template PATH/chat_template.jinja
"""
import argparse
import json
import sys
from pathlib import Path

CASES = {
    "un turno utente": {
        "messages": [{"role": "user", "content": "ciao"}],
    },
    "sistema piu' utente": {
        "messages": [{"role": "system", "content": "Sei conciso."},
                     {"role": "user", "content": "capitale della Francia?"}],
    },
    "assistant in cronologia": {
        "messages": [{"role": "user", "content": "1+1?"},
                     {"role": "assistant", "content": "2"},
                     {"role": "user", "content": "e 2+2?"}],
    },
}

THINKING = (True, False)


def reference(template_text, *, messages, enable_thinking=True, add_generation_prompt=True):
    import jinja2

    def raise_exception(message):
        raise RuntimeError(message)

    environment = jinja2.Environment(trim_blocks=False, lstrip_blocks=False,
                                     extensions=["jinja2.ext.loopcontrols"])
    environment.filters["tojson"] = (
        lambda value, ensure_ascii=False, **kw: json.dumps(value, ensure_ascii=ensure_ascii))
    environment.globals["raise_exception"] = raise_exception
    rendered = environment.from_string(template_text)
    return rendered.render(messages=messages, add_generation_prompt=add_generation_prompt,
                           enable_thinking=enable_thinking)


def show(label, ours, theirs):
    print(f"FAIL {label}")
    for index, (a, b) in enumerate(zip(ours.splitlines(), theirs.splitlines())):
        if a != b:
            print(f"  prima differenza alla riga {index + 1}")
            print(f"    gateway:  {a!r}")
            print(f"    template: {b!r}")
            return
    print(f"  lunghezze diverse: gateway {len(ours)}, template {len(theirs)}")
    print(f"    coda gateway:  {ours[-120:]!r}")
    print(f"    coda template: {theirs[-120:]!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    arguments = parser.parse_args()

    if not arguments.template.exists():
        print(f"SKIP: manca {arguments.template}; il riferimento non c'e' e "
              f"questo test non ha verificato nulla")
        return 2                                  # salto != successo (vedi docstring)
    try:
        import jinja2                              # noqa: F401
    except ImportError:
        print("SKIP: jinja2 non installato; senza non c'e' riferimento")
        return 2                                  # salto != successo (vedi docstring)

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import openai_server

    openai_server.ARCH = "qwen36"
    template_text = arguments.template.read_text(encoding="utf-8")

    failures = 0
    for enable_thinking in THINKING:
        for label, case in CASES.items():
            name = f"{label} [thinking={enable_thinking}]"
            theirs = reference(template_text, messages=case["messages"],
                               enable_thinking=enable_thinking)
            ours = openai_server.render_chat_qwen(case["messages"],
                                                  enable_thinking=enable_thinking)
            if ours == theirs:
                print(f"ok   {name}")
            else:
                show(name, ours, theirs)
                failures += 1

    # Prosecuzione: l'ultimo turno assistant e' da CONTINUARE. Come qwen38, ChatML chiude
    # ogni turno con <|im_end|> e il template non ha un ramo di continuazione, quindi la
    # forma aperta e' il suo add_generation_prompt=False MENO il <|im_end|>\n finale. Qui in
    # piu' il turno da proseguire e' DOPO l'ultima domanda, e li' il template scrive il blocco
    # <think></think> (una cronologia piu' vecchia lo perde): la resa aperta lo tiene.
    aperto = [{"role": "user", "content": "capitale della Francia?"},
              {"role": "assistant", "content": "La capitale e'"}]
    produced = openai_server.render_chat_for_arch(aperto, enable_thinking=True,
                                                  add_generation_prompt=False)
    closed = reference(template_text, messages=aperto, add_generation_prompt=False)
    # Il terminatore da togliere DEVE esserci nel riferimento: toglierlo "se c'e'" sarebbe un
    # no-op silenzioso se il template cambiasse convenzione, e il confronto perderebbe senso.
    TERM = "<|im_end|>\n"
    if not closed.endswith(TERM):
        print(f"FAIL prosecuzione: il riferimento non finisce col terminatore {TERM!r} da "
              f"togliere -- convenzione del template cambiata? coda: {closed[-40:]!r}")
        failures += 1
        expected = closed
    else:
        expected = closed[:-len(TERM)]
    if produced == expected:
        print("ok   prosecuzione: turno aperto = template(add_generation_prompt=False) "
              "senza il <|im_end|> finale")
    else:
        show("prosecuzione", produced, expected)
        failures += 1
    if not produced.endswith("La capitale e'"):
        print(f"FAIL prosecuzione: il prompt non finisce sull'apertura del client: "
              f"{produced[-60:]!r}")
        failures += 1
    if produced.rstrip("\n").endswith("<|im_end|>"):
        print("FAIL prosecuzione: il turno resta chiuso col terminatore")
        failures += 1
    # Controllo negativo: col ramo normale lo stesso scambio DEVE finire sulla cue.
    if not openai_server.render_chat_for_arch(
            aperto, enable_thinking=True).endswith("<think>\n"):
        print("FAIL prosecuzione: il ramo normale non emette piu' il prompt di generazione")
        failures += 1

    print()
    if failures:
        print(f"TEST FAIL ({failures} casi)")
        return 1
    print("template Qwen3.6: il gateway e' identico al riferimento")
    return 0


if __name__ == "__main__":
    sys.exit(main())
