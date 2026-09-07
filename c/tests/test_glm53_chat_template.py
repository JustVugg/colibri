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

    # Il ragionamento "spento" non e' una forma nostra: e' il livello minimo del
    # template. GLM-5.3 non sa non ragionare -- `effective_reasoning_effort` ha un
    # ramo else che vale 'max' e non e' mai none, e il prompt di generazione apre
    # sempre <think>. Quindi enable_thinking=False deve rendere ESATTAMENTE quello
    # che il template rende con reasoning_effort='low', e questo lo pretende
    # confrontando i due, non una forma scritta a mano qui.
    #
    # La versione precedente di questo blocco pretendeva l'opposto: nessuna riga di
    # effort e <think></think> chiuso. Era la mia teoria di #1282, misurata falsa e
    # ritirata su #1278, e il test la teneva in vita: due deviazioni dal template
    # trasformate in contratto. Un test puo' proteggere un errore tanto quanto una
    # correttezza -- per questo qui il riferimento e' il template, mai la mia idea
    # di cosa dovrebbe uscire.
    off = openai_server.render_chat_for_arch(
        [{"role": "user", "content": "ciao"}], enable_thinking=False)
    expected_off = reference(template_text,
                             messages=[{"role": "user", "content": "ciao"}],
                             reasoning_effort="low")
    if off != expected_off:
        print("FAIL: col ragionamento al minimo il gateway non rende il template")
        print(f"  gateway:  {off!r}")
        print(f"  template: {expected_off!r}")
        return 1
    if off.endswith("<think></think>"):
        print("FAIL: e' tornata la forma col blocco chiuso, che il modello non ha "
              "mai visto in questa posizione (#1278)")
        return 1
    if "Reasoning Effort" not in off:
        print("FAIL: manca la riga di effort, che il template emette sempre (#1278)")
        return 1
    checked += 1

    print(f"PASS GLM-5.3 chat template: {checked} rese identiche a "
          f"chat_template.jinja, strumenti e livelli di ragionamento compresi, "
          f"piu' il livello minimo, che rende come il template con effort low")
    return 0


if __name__ == "__main__":
    sys.exit(main())
