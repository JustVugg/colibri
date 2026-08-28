#!/usr/bin/env python3
"""Il renderer di GLM-5.3 nel gateway, contro chat_template.jinja ufficiale.

Il gateway rende i prompt a mano invece di far girare jinja a ogni richiesta, e
quella scelta si paga in un modo solo: la copia scritta a mano puo' scostarsi
dall'originale senza che nessuno se ne accorga, perche' il modello risponde
comunque. Un a capo di troppo dentro il blocco degli strumenti non da' errore,
da' risposte peggiori.

Qui il template vero viene reso con jinja2 e confrontato byte per byte con
quello che produce il gateway, sui casi che contano: senza strumenti, con uno,
con due, con una chiamata e il suo risultato, e con i livelli di ragionamento.

Serve chat_template.jinja del checkpoint. Se non c'e' il test si dichiara
saltato invece di passare: un test che non ha trovato il suo riferimento non ha
verificato niente, e dirlo verde sarebbe peggio che non averlo.

USO:
  python3 tests/test_glm53_chat_template.py --template PATH/chat_template.jinja
"""
import argparse
import json
import sys
from pathlib import Path

CASES = {
    "senza strumenti": {
        "messages": [{"role": "user", "content": "ciao"}],
    },
    "un solo strumento": {
        "messages": [{"role": "user", "content": "che tempo fa a Roma?"}],
        "tools": [{"type": "function", "function": {
            "name": "meteo", "description": "Il tempo",
            "parameters": {"type": "object",
                           "properties": {"citta": {"type": "string"}},
                           "required": ["citta"]}}}],
    },
    "due strumenti": {
        "messages": [{"role": "user", "content": "x"}],
        "tools": [
            {"type": "function", "function": {"name": "meteo", "description": "Il tempo"}},
            {"type": "function", "function": {"name": "ora", "description": "L'ora"}},
        ],
    },
    "chiamata e risultato": {
        "messages": [
            {"role": "user", "content": "che tempo fa a Roma?"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"type": "function", "function": {
                    "name": "meteo", "arguments": {"citta": "Roma"}}}]},
            {"role": "tool", "content": "sereno, 24 gradi"},
        ],
        "tools": [{"type": "function", "function": {
            "name": "meteo", "description": "Il tempo"}}],
    },
    "turno precedente con ragionamento": {
        "messages": [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "<think>rifletto</think>b"},
            {"role": "user", "content": "c"},
        ],
    },
    "istruzione di sistema": {
        "messages": [
            {"role": "system", "content": "sii breve"},
            {"role": "user", "content": "ciao"},
        ],
    },
}

EFFORTS = {"low": "low", "high": "high", "xhigh": None, None: None}


def reference(template_text, *, messages, tools=None, reasoning_effort=None):
    import jinja2
    environment = jinja2.Environment(trim_blocks=False, lstrip_blocks=False,
                                     extensions=["jinja2.ext.loopcontrols"])
    # jinja2 3.x non accetta ensure_ascii sul tojson incorporato; il template
    # lo passa esplicitamente, quindi il filtro va sostituito con uno che lo
    # capisce. Non cambia l'uscita, cambia solo la firma.
    environment.filters["tojson"] = (
        lambda value, ensure_ascii=False, **kw: json.dumps(value, ensure_ascii=ensure_ascii))
    rendered = environment.from_string(template_text)
    arguments = {"messages": messages, "add_generation_prompt": True}
    if tools:
        arguments["tools"] = tools
    if reasoning_effort:
        arguments["reasoning_effort"] = reasoning_effort
    return rendered.render(**arguments)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    arguments = parser.parse_args()

    if not arguments.template.exists():
        print(f"SKIP: manca {arguments.template}; il riferimento non c'e' e "
              f"questo test non ha verificato nulla")
        return 0
    try:
        import jinja2                              # noqa: F401
    except ImportError:
        print("SKIP: jinja2 non installato; senza non c'e' riferimento")
        return 0

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import openai_server

    openai_server.ARCH = "glm53"
    template_text = arguments.template.read_text(encoding="utf-8")
    checked = 0
    for name, case in CASES.items():
        for effort, passed in EFFORTS.items():
            expected = reference(template_text, messages=case["messages"],
                                 tools=case.get("tools"), reasoning_effort=passed)
            # Il confronto col template e' sul ragionamento ACCESO, che e'
            # l'unica forma che il template conosce: per lui il modello ragiona
            # sempre e il prompt di generazione apre <think>.
            produced = openai_server.render_chat_for_arch(
                case["messages"], enable_thinking=True,
                reasoning_effort=effort, tools=case.get("tools"))
            if produced != expected:
                print(f"FAIL {name} (reasoning_effort={effort!r})")
                for position, (left, right) in enumerate(zip(produced, expected)):
                    if left != right:
                        start = max(0, position - 40)
                        print(f"  primo scostamento a {position}")
                        print(f"  gateway:  ...{produced[start:position + 40]!r}")
                        print(f"  template: ...{expected[start:position + 40]!r}")
                        break
                else:
                    print(f"  lunghezze diverse: {len(produced)} contro {len(expected)}")
                return 1
            checked += 1

    # Il ragionamento spento e' una nostra aggiunta: il template non ha quella
    # forma perche' non prevede di spegnerlo. Non e' pero' inventata -- e'
    # quello che il template stesso scrive davanti a un turno passato senza
    # ragionamento -- e va comunque verificata, perche' l'unica cosa che la
    # tiene giusta e' questo controllo.
    off = openai_server.render_chat_for_arch(
        [{"role": "user", "content": "ciao"}], enable_thinking=False)
    on = openai_server.render_chat_for_arch(
        [{"role": "user", "content": "ciao"}], enable_thinking=True)
    if not off.endswith("<|assistant|><think></think>"):
        print(f"FAIL: col ragionamento spento il prompt finisce con {off[-40:]!r}")
        return 1
    if off[:-len("</think>")] != on:
        print("FAIL: acceso e spento differiscono per piu' della chiusura del blocco")
        return 1

    print(f"PASS GLM-5.3 chat template: {checked} rese identiche a "
          f"chat_template.jinja, strumenti e livelli di ragionamento compresi, "
          f"piu' la forma col ragionamento spento")
    return 0


if __name__ == "__main__":
    sys.exit(main())
