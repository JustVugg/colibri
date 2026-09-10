#!/usr/bin/env python3
"""Il renderer base di GLM-5.2 nel gateway, contro chat_template.jinja ufficiale.

Il gateway rende i prompt a mano invece di far girare jinja a ogni richiesta, e la
copia scritta a mano puo' scostarsi dall'originale senza che nessuno se ne accorga.
Qui il template vero viene reso con jinja2 e confrontato byte per byte.

I casi girano con enable_thinking=False di proposito. Con il ragionamento acceso il
renderer NON e' byte-identico al template, e apposta: la riga `Reasoning Effort` del
template mappa ogni livello che non sia `high` su `Max` (non monotono, #809), mentre il
gateway mappa minimal/low/medium/high/xhigh su Low/Low/Medium/High/Max. Quella
divergenza e' precedente a questo lavoro e non riguarda la prosecuzione: il turno aperto
e' identico con o senza ragionamento, perche' dipende solo dalla forma del turno passato,
non dalla cue.

Se manca il template o jinja2, il test si dichiara SALTATO invece di passare: un
test che non ha trovato il suo riferimento non ha verificato niente, e dirlo
verde sarebbe peggio -- percio' un salto esce con codice 2, distinto dallo 0 di
un confronto riuscito.

RIFERIMENTO (scaricato 2026-09-10):
  repo     zai-org/GLM-5.2-FP8
  file     chat_template.jinja
  sha256   172dc74a35e1752df75ecfb2b2cf9326d2852bb1379868ebeec9571654489679
  hf download zai-org/GLM-5.2-FP8 chat_template.jinja

USO:
  python3 tests/test_glm52_chat_template.py --template PATH/chat_template.jinja
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


def reference(template_text, *, messages, add_generation_prompt=True):
    import jinja2
    environment = jinja2.Environment(trim_blocks=False, lstrip_blocks=False,
                                     extensions=["jinja2.ext.loopcontrols"])
    environment.filters["tojson"] = (
        lambda value, ensure_ascii=False, **kw: json.dumps(value, ensure_ascii=ensure_ascii))
    rendered = environment.from_string(template_text)
    return rendered.render(messages=messages, add_generation_prompt=add_generation_prompt,
                           enable_thinking=False)


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

    openai_server.ARCH = "glm"
    template_text = arguments.template.read_text(encoding="utf-8")

    failures = 0
    for label, case in CASES.items():
        theirs = reference(template_text, messages=case["messages"])
        ours = openai_server.render_chat(case["messages"], enable_thinking=False)
        if ours == theirs:
            print(f"ok   {label}")
        else:
            show(label, ours, theirs)
            failures += 1

    # Prosecuzione: l'ultimo turno assistant e' da CONTINUARE. GLM non ha un terminatore di
    # turno (e' il token di ruolo seguente a chiudere), quindi la forma aperta e' il template
    # con add_generation_prompt=False cosi' com'e' -- niente da togliere, a differenza di
    # ChatML. Il prompt finisce su <|assistant|><think></think>{contenuto}.
    aperto = [{"role": "user", "content": "capitale della Francia?"},
              {"role": "assistant", "content": "La capitale e'"}]
    produced = openai_server.render_chat_for_arch(aperto, enable_thinking=False,
                                                  add_generation_prompt=False)
    expected = reference(template_text, messages=aperto, add_generation_prompt=False)
    if produced == expected:
        print("ok   prosecuzione: turno aperto = template(add_generation_prompt=False)")
    else:
        show("prosecuzione", produced, expected)
        failures += 1
    if not produced.endswith("La capitale e'"):
        print(f"FAIL prosecuzione: il prompt non finisce sull'apertura del client: "
              f"{produced[-60:]!r}")
        failures += 1
    # Controllo negativo: col ramo normale lo stesso scambio DEVE finire sulla cue.
    if not openai_server.render_chat_for_arch(
            aperto, enable_thinking=False).endswith("<|assistant|><think></think>"):
        print("FAIL prosecuzione: il ramo normale non emette piu' il prompt di generazione")
        failures += 1

    print()
    if failures:
        print(f"TEST FAIL ({failures} casi)")
        return 1
    print("template GLM-5.2: il gateway e' identico al riferimento (ragionamento spento)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
