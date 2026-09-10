#!/usr/bin/env python3
"""Il renderer di OLMoE nel gateway, contro il chat_template del checkpoint.

Il template di OLMoE vive dentro tokenizer_config.json, non in un chat_template.jinja
a se'. Estrai il campo `chat_template` in un file e passalo qui. bos_token ed eos_token
sono lo stesso marcatore, "|||IP_ADDRESS|||".

Se manca il template o jinja2, il test si dichiara SALTATO invece di passare: un
test che non ha trovato il suo riferimento non ha verificato niente, e dirlo
verde sarebbe peggio -- percio' un salto esce con codice 2, distinto dallo 0 di
un confronto riuscito.

RIFERIMENTO (scaricato 2026-09-10):
  repo     allenai/OLMoE-1B-7B-0125-Instruct
  file     campo `chat_template` estratto da tokenizer_config.json
  sha256   fe689ffbd6a4e2d0532d7480696b065b10e0e1eff3f9b9fc4bea415761e4bf4a  (del .jinja estratto)
  hf download allenai/OLMoE-1B-7B-0125-Instruct tokenizer_config.json  # poi estrai chat_template

USO:
  python3 tests/test_olmoe_chat_template.py --template PATH/olmoe-chat_template.jinja
"""
import argparse
import sys
from pathlib import Path

BOUNDARY = "|||IP_ADDRESS|||"   # bos_token == eos_token

CASES = {
    "un turno utente": [{"role": "user", "content": "ciao"}],
    "sistema piu' utente": [{"role": "system", "content": "Sei conciso."},
                            {"role": "user", "content": "capitale della Francia?"}],
    "assistant in cronologia": [{"role": "user", "content": "1+1?"},
                                {"role": "assistant", "content": "2"},
                                {"role": "user", "content": "e 2+2?"}],
}


def reference(template_text, *, messages, add_generation_prompt=True):
    import jinja2
    environment = jinja2.Environment(trim_blocks=False, lstrip_blocks=False,
                                     extensions=["jinja2.ext.loopcontrols"])
    rendered = environment.from_string(template_text)
    return rendered.render(messages=messages, add_generation_prompt=add_generation_prompt,
                           bos_token=BOUNDARY, eos_token=BOUNDARY)


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

    openai_server.ARCH = "olmoe"
    template_text = arguments.template.read_text(encoding="utf-8")

    failures = 0
    for label, messages in CASES.items():
        theirs = reference(template_text, messages=messages)
        ours = openai_server.render_chat_olmoe(messages)
        if ours == theirs:
            print(f"ok   {label}")
        else:
            show(label, ours, theirs)
            failures += 1

    # Prosecuzione: l'ultimo turno assistant e' da CONTINUARE. Il template chiude anche
    # l'ultimo turno con eos_token, quindi la forma aperta e' quel rendering MENO l'eos
    # finale e senza cue -- lo stesso taglio del terminatore delle famiglie ChatML.
    aperto = [{"role": "user", "content": "capitale della Francia?"},
              {"role": "assistant", "content": "La capitale e'"}]
    produced = openai_server.render_chat_for_arch(aperto, add_generation_prompt=False)
    closed = reference(template_text, messages=aperto, add_generation_prompt=False)
    # L'eos da togliere DEVE esserci nel riferimento: toglierlo "se c'e'" sarebbe un no-op
    # silenzioso se il template cambiasse convenzione, e il confronto perderebbe senso.
    if not closed.endswith(BOUNDARY):
        print(f"FAIL prosecuzione: il riferimento non finisce con l'eos {BOUNDARY!r} da "
              f"togliere -- convenzione del template cambiata? coda: {closed[-40:]!r}")
        failures += 1
        expected = closed
    else:
        expected = closed[:-len(BOUNDARY)]
    if produced == expected:
        print("ok   prosecuzione: turno aperto = template(add_generation_prompt=False) "
              "senza l'eos finale")
    else:
        show("prosecuzione", produced, expected)
        failures += 1
    if not produced.endswith("La capitale e'"):
        print(f"FAIL prosecuzione: il prompt non finisce sull'apertura del client: "
              f"{produced[-60:]!r}")
        failures += 1
    if produced.endswith(BOUNDARY):
        print("FAIL prosecuzione: il turno resta chiuso con l'eos")
        failures += 1
    # Controllo negativo: col ramo normale lo stesso scambio DEVE finire sulla cue.
    if not openai_server.render_chat_for_arch(aperto).endswith("<|assistant|>\n"):
        print("FAIL prosecuzione: il ramo normale non emette piu' il prompt di generazione")
        failures += 1

    print()
    if failures:
        print(f"TEST FAIL ({failures} casi)")
        return 1
    print("template OLMoE: il gateway e' identico al riferimento")
    return 0


if __name__ == "__main__":
    sys.exit(main())
