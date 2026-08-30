#!/usr/bin/env python3
"""Il renderer di Qwen3.8 nel gateway, contro chat_template.jinja ufficiale.

Il gateway rende i prompt a mano invece di far girare jinja a ogni richiesta.
Quella scelta si paga in un modo solo: la copia scritta a mano puo' scostarsi
dall'originale senza che nessuno se ne accorga, perche' il modello risponde
comunque. Qui pesa piu' che altrove, perche' il blocco degli strumenti di
Qwen3.8 non e' JSON ma un formato suo:

    <tool_call>
    <function=NOME>
    <parameter=CHIAVE>
    VALORE
    </parameter>
    </function>
    </tool_call>

ed e' la dichiarazione stessa a insegnare al modello la sintassi che deve
emettere. Un preambolo parafrasato e' un preambolo che il modello non ha mai
visto: non da' errore, da' chiamate malformate.

Il template vero viene reso con jinja2 e confrontato byte per byte con quello
che produce il gateway. Se manca il template il test si dichiara SALTATO invece
di passare: un test che non ha trovato il suo riferimento non ha verificato
niente, e dirlo verde sarebbe peggio che non averlo.

USO:
  python3 tests/test_qwen38_chat_template.py --template PATH/chat_template.jinja
"""
import argparse
import json
import sys
from pathlib import Path

METEO = {"type": "function", "function": {
    "name": "meteo", "description": "Il tempo che fa",
    "parameters": {"type": "object",
                   "properties": {"citta": {"type": "string"},
                                  "giorni": {"type": "integer"}},
                   "required": ["citta"]}}}
ORA = {"type": "function", "function": {"name": "ora", "description": "L'ora corrente"}}

CASES = {
    "senza strumenti": {
        "messages": [{"role": "user", "content": "ciao"}],
    },
    "un solo strumento": {
        "messages": [{"role": "user", "content": "che tempo fa a Roma?"}],
        "tools": [METEO],
    },
    "due strumenti": {
        "messages": [{"role": "user", "content": "x"}],
        "tools": [METEO, ORA],
    },
    "strumenti piu' messaggio di sistema": {
        "messages": [{"role": "system", "content": "sii breve"},
                     {"role": "user", "content": "che ora e'?"}],
        "tools": [ORA],
    },
    "chiamata senza testo che la precede": {
        "messages": [
            {"role": "user", "content": "che tempo fa a Roma?"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"type": "function", "function": {
                    "name": "meteo", "arguments": {"citta": "Roma"}}}]},
            {"role": "tool", "content": "sereno, 24 gradi"},
            {"role": "user", "content": "e domani?"},
        ],
        "tools": [METEO],
    },
    "chiamata preceduta da testo": {
        "messages": [
            {"role": "user", "content": "che tempo fa a Roma?"},
            {"role": "assistant", "content": "Controllo subito.", "tool_calls": [
                {"type": "function", "function": {
                    "name": "meteo", "arguments": {"citta": "Roma"}}}]},
            {"role": "tool", "content": "sereno"},
            {"role": "user", "content": "grazie"},
        ],
        "tools": [METEO],
    },
    "due chiamate nello stesso turno": {
        "messages": [
            {"role": "user", "content": "meteo e ora"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"type": "function", "function": {
                    "name": "meteo", "arguments": {"citta": "Roma"}}},
                {"type": "function", "function": {"name": "ora", "arguments": {}}}]},
            {"role": "tool", "content": "sereno"},
            {"role": "tool", "content": "14:30"},
            {"role": "user", "content": "ok"},
        ],
        "tools": [METEO, ORA],
    },
    "argomento non stringa": {
        "messages": [
            {"role": "user", "content": "meteo Roma tre giorni"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"type": "function", "function": {
                    "name": "meteo", "arguments": {"citta": "Roma", "giorni": 3}}}]},
            {"role": "tool", "content": "sereno"},
            {"role": "user", "content": "ok"},
        ],
        "tools": [METEO],
    },
    "turno precedente con ragionamento": {
        "messages": [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b", "reasoning_content": "rifletto"},
            {"role": "user", "content": "c"},
        ],
    },
}

EFFORTS = ("xhigh", "medium", "low")


def reference(template_text, *, messages, tools=None, reasoning_effort=None):
    import jinja2

    def raise_exception(message):
        raise RuntimeError(message)

    environment = jinja2.Environment(trim_blocks=False, lstrip_blocks=False,
                                     extensions=["jinja2.ext.loopcontrols"])
    environment.filters["tojson"] = (
        lambda value, ensure_ascii=False, **kw: json.dumps(value, ensure_ascii=ensure_ascii))
    environment.globals["raise_exception"] = raise_exception
    rendered = environment.from_string(template_text)
    arguments = {"messages": messages, "add_generation_prompt": True,
                 "enable_thinking": True}
    if tools:
        arguments["tools"] = tools
    if reasoning_effort:
        arguments["reasoning_effort"] = reasoning_effort
    return rendered.render(**arguments)


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
        return 0
    try:
        import jinja2                              # noqa: F401
    except ImportError:
        print("SKIP: jinja2 non installato; senza non c'e' riferimento")
        return 0

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import openai_server

    openai_server.ARCH = "qwen38"
    template_text = arguments.template.read_text(encoding="utf-8")

    failures = 0
    for effort in EFFORTS:
        for label, case in CASES.items():
            name = f"{label} [{effort}]"
            theirs = reference(template_text, messages=case["messages"],
                               tools=case.get("tools"), reasoning_effort=effort)
            ours = openai_server.render_chat_qwen38(
                case["messages"], enable_thinking=True, reasoning_effort=effort,
                tools=case.get("tools"))
            if ours == theirs:
                print(f"ok   {name}")
            else:
                show(name, ours, theirs)
                failures += 1

    # Il giro completo: rendere una chiamata e rileggerla deve restituire quello
    # che ci era stato dato. E' la meta' che il confronto col template non copre,
    # perche' il template sa solo scrivere.
    reply = ("Controllo.\n\n<tool_call>\n<function=meteo>\n<parameter=citta>\n"
             "Roma\n</parameter>\n<parameter=giorni>\n3\n</parameter>\n"
             "</function>\n</tool_call>")
    text, calls = openai_server.parse_qwen38_tool_calls(reply, [METEO])
    expected = {"citta": "Roma", "giorni": 3}
    got = json.loads(calls[0]["function"]["arguments"]) if calls else None
    if len(calls) == 1 and calls[0]["function"]["name"] == "meteo" and got == expected \
            and text == "Controllo.":
        print("ok   andata e ritorno: chiamata riletta, interi restituiti come interi")
    else:
        print(f"FAIL andata e ritorno: {calls!r} testo={text!r}")
        failures += 1

    print()
    if failures:
        print(f"TEST FAIL ({failures} casi)")
        return 1
    print("template Qwen3.8: il gateway e' identico al riferimento")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
