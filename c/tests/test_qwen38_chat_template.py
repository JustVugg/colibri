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
niente, e dirlo verde sarebbe peggio che non averlo -- percio' un salto esce con
codice 2, distinto dallo 0 di un confronto riuscito.

RIFERIMENTO (scaricato 2026-09-10):
  repo     Qwen/Qwen3.8-Flash-Next-FP8
  file     chat_template.jinja
  sha256   c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041
  hf download Qwen/Qwen3.8-Flash-Next-FP8 chat_template.jinja

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


def reference(template_text, *, messages, tools=None, reasoning_effort=None,
              add_generation_prompt=True):
    import jinja2

    def raise_exception(message):
        raise RuntimeError(message)

    environment = jinja2.Environment(trim_blocks=False, lstrip_blocks=False,
                                     extensions=["jinja2.ext.loopcontrols"])
    environment.filters["tojson"] = (
        lambda value, ensure_ascii=False, **kw: json.dumps(value, ensure_ascii=ensure_ascii))
    environment.globals["raise_exception"] = raise_exception
    rendered = environment.from_string(template_text)
    arguments = {"messages": messages, "add_generation_prompt": add_generation_prompt,
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
        return 2                                  # salto != successo (vedi docstring)
    try:
        import jinja2                              # noqa: F401
    except ImportError:
        print("SKIP: jinja2 non installato; senza non c'e' riferimento")
        return 2                                  # salto != successo (vedi docstring)

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

    # Prosecuzione: l'ultimo turno assistant e' da CONTINUARE, non uno gia' finito.
    # ChatML chiude ogni turno con <|im_end|>, e questo template non ha un ramo di
    # continuazione: il suo add_generation_prompt=False toglie solo la cue, il turno resta
    # chiuso. La forma aperta e' percio' quel rendering MENO il <|im_end|>\n finale -- la
    # posizione in cui il modello si trova mentre scrive un turno, e da cui prosegue.
    aperto = [{"role": "user", "content": "capitale della Francia?"},
              {"role": "assistant", "content": "La capitale e'"}]
    produced = openai_server.render_chat_for_arch(aperto, enable_thinking=True,
                                                  reasoning_effort="low",
                                                  add_generation_prompt=False)
    closed = reference(template_text, messages=aperto, reasoning_effort="low",
                       add_generation_prompt=False)
    # Il terminatore da togliere DEVE esserci nel riferimento: se il template cambiasse
    # convenzione, toglierlo "se c'e'" sarebbe un no-op silenzioso e il confronto perderebbe
    # senso. Percio' lo esigiamo prima di tagliarlo.
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
    # Controllo negativo: col ramo normale lo stesso scambio DEVE finire sulla cue, o il
    # confronto qui sopra non starebbe distinguendo niente.
    if not openai_server.render_chat_for_arch(
            aperto, enable_thinking=True, reasoning_effort="low").endswith("<think>\n"):
        print("FAIL prosecuzione: il ramo normale non emette piu' il prompt di generazione")
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
