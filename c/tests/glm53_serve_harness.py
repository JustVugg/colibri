#!/usr/bin/env python3
"""Il protocollo di serve del motore GLM-5.3, parlato per intero.

`coli chat`, `coli serve` e `coli web` non chiamano la CLI: aprono una pipa e
parlano il protocollo di docs/serve_protocol.md. Fra la CLI che funziona e il
server che funziona ci sono l'handshake, il frame a byte contati, la
tokenizzazione del payload, la decodifica dei token in uscita e i codici di
errore, e nessuna di queste cose e' coperta da un test sui token.

Il confronto e' con la CLI sullo stesso prompt: la stessa domanda posta nei due
modi deve dare la stessa risposta, altrimenti la differenza sta nel protocollo.

Si controllano anche i rifiuti, perche' un server che accetta un frame rotto
invece di dirlo si disallinea sullo stream e da li' in poi risponde a domande
che nessuno ha fatto.
"""
import argparse
import os
import tempfile
import time
import subprocess
import sys
from pathlib import Path


NOTES = tempfile.mktemp(suffix=".glm53.stderr")


def reuse_reported():
    """Quanti token di prefisso il motore dice di aver riusato."""
    try:
        text = open(NOTES, "r", errors="replace").read()
    except OSError:
        return 0
    return sum(int(line.split()[2]) for line in text.splitlines()
               if line.startswith("REUSE "))


def reuse_line(request_id):
    """La riga REUSE di una richiesta, spezzata in campi (vuota se non c'e')."""
    try:
        text = open(NOTES, "r", errors="replace").read()
    except OSError:
        return []
    for line in text.splitlines():
        if line.startswith(f"REUSE {request_id} "):
            return line.split()
    return []


def engine(binary, fixture, extra=None):
    # GLM53_VERBOSE: il riuso del prefisso si racconta solo su richiesta,
    # perche' `coli chat` eredita lo stderr del server e quella riga finirebbe
    # a schermo dopo ogni risposta.
    environment = {**os.environ, "SERVE": "1", "SERVE_BATCH": "1",
                   "SNAP": str(fixture), "GLM53_BITS": "32", "GLM53_VERBOSE": "1"}
    environment.update(extra or {})
    # stderr in un file: il riuso del prefisso si racconta li', perche' nel
    # protocollo una riga in piu' farebbe cadere il gateway (di proposito).
    return subprocess.Popen([binary], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=open(NOTES, "wb"), env=environment)


def read_line(stream):
    line = stream.readline()
    if not line:
        raise AssertionError("il motore ha chiuso lo stream prima del previsto")
    return line.decode("utf-8", "replace").rstrip("\n")


def handshake(process):
    ready = read_line(process.stdout)
    if "READY" not in ready:
        raise AssertionError(f"prima riga {ready!r}, atteso il sentinello READY")
    stat = read_line(process.stdout)
    while stat.startswith("CAPS "):        # CAPS vision=<0|1> sits between READY and STAT
        stat = read_line(process.stdout)
    if not stat.startswith("STAT "):
        raise AssertionError(f"seconda riga {stat!r}, atteso STAT")


def submit_bytes(process, request_id, body, max_tokens=4, temperature=0.0, top_p=1.0):
    header = (f"SUBMIT {request_id} 0 {len(body)} {max_tokens} "
              f"{temperature} {top_p}\n").encode("utf-8")
    process.stdin.write(header + body + b"\n")
    process.stdin.flush()


def submit(process, request_id, payload, max_tokens=4, temperature=0.0, top_p=1.0):
    submit_bytes(process, request_id, payload.encode("utf-8"), max_tokens,
                 temperature, top_p)


def collect(process, request_id):
    """I DATA fino al DONE della richiesta, o l'ERROR che li sostituisce.

    Restituisce anche quanto prefisso lo slot ha riusato, se il motore lo dice."""
    pieces = []
    while True:
        line = read_line(process.stdout)
        if line.startswith("DATA "):
            _, got_id, count = line.split()
            if int(got_id) != request_id:
                raise AssertionError(f"DATA per {got_id}, atteso {request_id}")
            payload = process.stdout.read(int(count) + 1)      # payload piu' '\n'
            pieces.append(payload[:int(count)])
        elif line.startswith("DONE ") or line.startswith("ERROR "):
            return b"".join(pieces), line, reuse_reported()


def fresh_answer(binary, fixture, body, tokens, extra=None):
    """La risposta di un motore appena partito: il riferimento per ogni riuso.
    Un motore suo, perche' lo slot di un altro potrebbe riusare qualcosa."""
    reference = engine(binary, fixture, extra)
    try:
        handshake(reference)
        submit_bytes(reference, 1, body, max_tokens=tokens)
        answer, done, _ = collect(reference, 1)
        reference.stdin.close()
        reference.wait(timeout=60)
    finally:
        if reference.poll() is None:
            reference.kill()
    if not done.startswith("DONE 1 "):
        raise AssertionError(f"il riferimento ha chiuso con {done!r}")
    return answer


def cli_answer(binary, fixture, prompt, tokens):
    """L'uscita della CLI in BYTE.

    Un tokenizzatore a livello di byte emette byte grezzi, e un carattere
    multibyte si spezza fra due token: la risposta di un modello a pesi casuali
    non e' UTF-8 valido ne' deve esserlo. Decodificarla qui vorrebbe dire
    rompere il test su un fatto normale del formato."""
    result = subprocess.run(
        [binary, "--model", str(fixture), "--prompt", prompt, "--greedy", str(tokens)],
        capture_output=True, check=True,
        env={**os.environ, "GLM53_BITS": "32"})
    return result.stdout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    arguments = parser.parse_args()
    if not (arguments.fixture / "ref.json").exists():
        # Un traceback su un file che manca fa sembrare rotto il
        # motore; chi arriva per la prima volta non puo' distinguere
        # le due cose. Il generatore vuole transformers 5.16.1.
        print(f"SKIP: manca {arguments.fixture}; generalo con\n"
              f"  python3 tools/make_glm53_multimodal_tiny.py --output <dir>")
        return 2  # un salto non e' un successo
    binary = os.path.abspath(arguments.binary)
    prompt, tokens = "gu", 4

    process = engine(binary, arguments.fixture)
    try:
        handshake(process)

        submit(process, 7, prompt, max_tokens=tokens)
        served, done, _ = collect(process, 7)
        if not done.startswith("DONE 7 STAT "):
            print(f"FAIL: chiusura {done!r}")
            return 1
        fields = done.split()
        if len(fields) != 9:
            print(f"FAIL: DONE con {len(fields)} campi invece di 9: {done!r}")
            return 1
        emitted = int(fields[3])
        if emitted != tokens:
            print(f"FAIL: {emitted} token emessi, chiesti {tokens}")
            return 1

        # Un prompt vuoto e' un rifiuto, non una risposta vuota.
        submit(process, 8, "", max_tokens=tokens)
        _, empty, _ = collect(process, 8)
        if empty != "ERROR 8 EMPTY_PROMPT":
            print(f"FAIL: prompt vuoto -> {empty!r}, atteso ERROR 8 EMPTY_PROMPT")
            return 1

        # Dopo un rifiuto lo stream deve restare allineato.
        submit(process, 9, prompt, max_tokens=1)
        after, done9, _ = collect(process, 9)
        if not done9.startswith("DONE 9 "):
            print(f"FAIL: dopo un errore lo stream si e' disallineato: {done9!r}")
            return 1
        if not served.startswith(after):
            print(f"FAIL: la stessa domanda ha dato due risposte diverse\n"
                  f"  4 token: {served!r}\n  1 token: {after!r}")
            return 1

        # --- slot KV: un secondo turno che estende il primo ---
        #
        # Il tokenizzatore e' a livello di byte senza merge, quindi tokenizzare
        # una concatenazione da' la concatenazione delle tokenizzazioni: il
        # prompt del secondo turno estende davvero la sequenza del primo, che e'
        # la condizione perche' lo slot possa riusarla.
        second = prompt.encode() + served + b"z"
        submit_bytes(process, 10, second, max_tokens=2)
        turn2, done10, reused = collect(process, 10)
        if not done10.startswith("DONE 10 "):
            print(f"FAIL: secondo turno {done10!r}")
            return 1
        if not reused:
            print("FAIL: lo slot non ha riusato niente su un prompt che estende "
                  "quello di prima")
            return 1
        if reuse_line(7)[-1:] != ["cold"] or reuse_line(10)[-1:] != ["extend"]:
            print(f"FAIL: motivi del riuso {reuse_line(7)!r} / {reuse_line(10)!r}, "
                  f"attesi cold e extend")
            return 1
        # Il turno 7 si e' fermato al limite di token: la storia dello slot ha
        # un token in piu' di quelli in cache, e il prompt del turno 10 combacia
        # anche con quello. <in comune> non deve superare <in cache>.
        if reuse_line(10)[5] != reuse_line(10)[4]:
            print(f"FAIL: REUSE 10 dice {reuse_line(10)[5]} in comune ma "
                  f"{reuse_line(10)[4]} in cache: {reuse_line(10)!r}")
            return 1

        # --- CANCEL a meta' turno (#1332) ---
        #
        # Un CANCEL che arriva mentre il turno gira deve fermarlo. Prima
        # serve_read_req era l'unico posto che leggeva il comando, e serve_loop
        # la chiama solo fra una richiesta e l'altra: il CANCEL restava nella
        # pipa fino alla fine dei token chiesti, e il gateway -- che manda
        # CANCEL e poi aspetta l'ack tenendo l'ammissione dello scheduler --
        # non liberava niente, quindi un client che si disconnetteva lasciava
        # le richieste successive in coda dietro una generazione che nessuno
        # voleva piu'.
        #
        # La prova che il turno e' stato interrotto davvero e' il numero di
        # DATA arrivati: molti meno dei max_tokens chiesti. Un motore che
        # leggesse il CANCEL solo a fine turno ne emetterebbe `budget`.
        budget = 512
        submit(process, 12, prompt, max_tokens=budget)
        emitted_before = 0
        cancelled_text = b""
        while True:
            line = read_line(process.stdout)
            if line.startswith("DONE ") or line.startswith("ERROR "):
                break
            if not line.startswith("DATA "):
                continue    # EMAP, HITS, PROF: si ignorano, come fa il gateway
            _, got_id, count = line.split()
            if int(got_id) != 12:
                raise AssertionError(f"DATA per {got_id}, atteso 12")
            cancelled_text += process.stdout.read(int(count) + 1)[:int(count)]
            emitted_before += 1
            if emitted_before == 1:
                process.stdin.write(b"CANCEL 12\n")
                process.stdin.flush()
        if line != "ERROR 12 CANCELLED":
            print(f"FAIL: CANCEL a meta' turno -> {line!r}, atteso "
                  f"ERROR 12 CANCELLED (il turno ha emesso {emitted_before} token)")
            return 1
        if emitted_before >= budget:
            print(f"FAIL: il turno ha emesso tutti i {budget} token chiesti prima "
                  f"di rispondere al CANCEL: non e' stato onorato a meta' turno")
            return 1

        # La riga CANCEL su stderr: e' l'unico modo di sapere, dopo, quanti
        # token il motore ha mandato e quanti ne tiene in cache. Qui la pipa
        # non perde niente, quindi `emitted` deve essere esattamente i DATA
        # letti; e `filled` deve essere prompt piu' emessi, che e' quello che
        # un Continue dovra' superare per riusare lo slot.
        notes = open(NOTES, "r", errors="replace").read().splitlines()
        cancel_lines = [line.split() for line in notes if line.startswith("CANCEL 12 ")]
        if len(cancel_lines) != 1 or len(cancel_lines[0]) != 5:
            print(f"FAIL: attesa una riga 'CANCEL 12 <prompt> <emessi> <filled>' "
                  f"su stderr, trovate {cancel_lines!r}")
            return 1
        cancel_prompt, cancel_emitted, cancel_filled = map(int, cancel_lines[0][2:])
        if cancel_emitted != emitted_before:
            print(f"FAIL: CANCEL dice {cancel_emitted} token emessi, ne sono "
                  f"arrivati {emitted_before}")
            return 1
        if cancel_filled != cancel_prompt + cancel_emitted:
            print(f"FAIL: CANCEL con filled {cancel_filled}, atteso prompt "
                  f"{cancel_prompt} + emessi {cancel_emitted}")
            return 1
        if not any(line.startswith("REUSE 12 ") for line in notes):
            print("FAIL: il turno interrotto non ha lasciato la sua riga REUSE")
            return 1

        # Il Continue di un client che ha ricevuto tutto: il prompt di prima
        # piu' esattamente i token arrivati. Coincide con la cache, token per
        # token, quindi non resta niente da macinare: il motore riprende dai
        # logit tenuti alla fine del turno interrotto, senza rifare il
        # prefill, e deve rispondere come un motore appena partito.
        continued = prompt.encode() + cancelled_text
        submit_bytes(process, 20, continued, max_tokens=1)
        resumed20, done20, _ = collect(process, 20)
        if not done20.startswith("DONE 20 "):
            print(f"FAIL: Continue dopo il CANCEL -> {done20!r}")
            return 1
        why20 = reuse_line(20)
        if why20[2:] != [str(cancel_filled)] * 4 + ["equal"]:
            print(f"FAIL: Continue identico alla cache -> REUSE {why20!r}, atteso "
                  f"{cancel_filled} riusati su {cancel_filled}, motivo equal")
            return 1
        if resumed20 != fresh_answer(binary, arguments.fixture, continued, 1):
            print(f"FAIL: ripreso dai logit tenuti, il Continue risponde "
                  f"{resumed20!r} e un motore appena partito no")
            return 1

        # Dopo un CANCEL lo stream deve restare allineato come dopo un errore:
        # il gateway riusa la stessa pipa per la richiesta successiva.
        submit(process, 13, prompt, max_tokens=1)
        _, done13, _ = collect(process, 13)
        if not done13.startswith("DONE 13 "):
            print(f"FAIL: dopo un CANCEL lo stream si e' disallineato: {done13!r}")
            return 1

        # --- STOP a meta' turno ---
        #
        # STOP non e' CANCEL, e i due non possono finire nello stesso posto. Il
        # protocollo: "STOP ends generation through the normal successful DONE
        # path. Statistics, usage history, and KV state are persisted". Il
        # gateway lo manda quando un template di stop combacia -- cioe' quando
        # la risposta e' completa e va consegnata -- e dopo averlo mandato
        # continua a leggere fino al DONE trattandolo come riuscito (alza
        # ClientCancelled solo se aveva mandato CANCEL). Rispondere
        # ERROR <id> CANCELLED butterebbe via una risposta che il client ha gia'
        # ricevuto per intero.
        stop_budget = 512
        submit(process, 14, prompt, max_tokens=stop_budget)
        emitted_stop = 0
        done14 = None
        while True:
            line = read_line(process.stdout)
            if line.startswith("DATA "):
                _, got_id, count = line.split()
                if int(got_id) != 14:
                    raise AssertionError(f"DATA per {got_id}, atteso 14")
                process.stdout.read(int(count) + 1)
                emitted_stop += 1
                if emitted_stop == 1:
                    process.stdin.write(b"STOP 14\n")
                    process.stdin.flush()
            elif line.startswith("DONE ") or line.startswith("ERROR "):
                done14 = line
                break
            # HITS, PROF e compagnia: si ignorano, come fa il gateway.
        if not done14.startswith("DONE 14 "):
            print(f"FAIL: STOP a meta' turno -> {done14!r}, atteso un DONE "
                  f"(il protocollo vuole il percorso riuscito, non CANCELLED)")
            return 1
        if emitted_stop >= stop_budget:
            print(f"FAIL: il turno ha emesso tutti i {stop_budget} token chiesti "
                  f"prima di rispondere allo STOP: non e' stato onorato a meta' "
                  f"turno")
            return 1

        # --- SUBMIT mentre lo slot e' occupato ---
        #
        # Non puo' legalmente arrivare -- il gateway serve una richiesta per
        # volta e aspetta il DONE -- ma se arriva non lo si tiene in coda: si
        # risponde SLOT_BUSY, che e' il codice che il protocollo documenta.
        # `BUSY` non esiste: un gateway che lo ricevesse non saprebbe che farsene
        # e il suo thread resterebbe appeso su una pipa che nessuno legge.
        #
        # La seconda meta' del controllo e' che il payload del SUBMIT rifiutato
        # venga comunque consumato: e' a byte contati, e lasciarlo nello stream
        # disallineerebbe tutto quello che segue.
        busy_budget = 8
        submit(process, 15, prompt, max_tokens=busy_budget)
        sent_intruder = False
        busy = None
        done15 = None
        while True:
            line = read_line(process.stdout)
            if line.startswith("DATA "):
                _, got_id, count = line.split()
                if int(got_id) != 15:
                    raise AssertionError(f"DATA per {got_id}, atteso 15")
                process.stdout.read(int(count) + 1)
                if not sent_intruder:
                    sent_intruder = True
                    submit(process, 16, prompt, max_tokens=1)
            elif line.startswith("ERROR "):
                busy = line
            elif line.startswith("DONE "):
                done15 = line
                break
        if busy != "ERROR 16 SLOT_BUSY":
            print(f"FAIL: SUBMIT a slot occupato -> {busy!r}, atteso "
                  f"ERROR 16 SLOT_BUSY")
            return 1
        if not done15.startswith("DONE 15 "):
            print(f"FAIL: il turno in volo e' finito con {done15!r}, atteso DONE 15")
            return 1

        # Il SUBMIT rifiutato non deve aver lasciato il suo payload nella pipa:
        # se ci fosse ancora, la richiesta dopo leggerebbe i suoi byte come
        # header e lo stream sarebbe perso.
        submit(process, 17, prompt, max_tokens=1)
        _, done17, _ = collect(process, 17)
        if not done17.startswith("DONE 17 "):
            print(f"FAIL: dopo un SUBMIT rifiutato lo stream si e' disallineato: "
                  f"{done17!r}")
            return 1

        process.stdin.close()
        process.wait(timeout=60)
    finally:
        if process.poll() is None:
            process.kill()

    # --- EOF su stdin a meta' turno ---
    #
    # "EOF on stdin = graceful shutdown: in-flight requests finish first." La
    # pipa che si chiude sotto un turno in volo non e' un motivo per troncare la
    # risposta: il turno finisce, il DONE parte, e solo dopo il motore esce.
    #
    # La prova non e' che il DONE arrivi -- arriverebbe anche troncando, perche'
    # il ciclo esce e il DONE e' la riga dopo -- ma che i token siano gli
    # STESSI di un turno con la pipa aperta. Greedy e a temperatura zero, quindi
    # lo stesso prompt deve dare la stessa sequenza: se il turno si fermasse al
    # primo token, qui si vedrebbe.
    def turn_with_eof(close_stdin):
        eof_process = engine(binary, arguments.fixture)
        try:
            handshake(eof_process)
            submit(eof_process, 19, prompt, max_tokens=8)
            if close_stdin:
                eof_process.stdin.close()
            pieces, done, _ = collect(eof_process, 19)
            if not done.startswith("DONE 19 "):
                return None, done
            if close_stdin:
                eof_process.wait(timeout=60)
            else:
                eof_process.stdin.close()
                eof_process.wait(timeout=60)
            return pieces, done
        finally:
            if eof_process.poll() is None:
                eof_process.kill()

    control, done_control = turn_with_eof(False)
    interrupted, done_interrupted = turn_with_eof(True)
    if control is None or interrupted is None:
        print(f"FAIL: EOF a meta' turno -> controllo {done_control!r}, "
              f"con la pipa chiusa {done_interrupted!r}, attesi due DONE")
        return 1
    if interrupted != control:
        print(f"FAIL: la pipa chiusa a meta' turno ha troncato la risposta\n"
              f"  pipa aperta: {control!r} ({len(control)} byte)\n"
              f"  pipa chiusa: {interrupted!r} ({len(interrupted)} byte)")
        return 1
    emitted_eof = len(control)

    # Lo stesso prompt su una sessione pulita: il riuso deve essere esatto, non
    # solo veloce. Se qui la risposta cambia, la cache tenuta fra i turni sta
    # dando al modello un contesto diverso da quello che il client crede.
    fresh = engine(binary, arguments.fixture)
    try:
        handshake(fresh)
        submit_bytes(fresh, 11, second, max_tokens=2)
        clean, done11, reused_clean = collect(fresh, 11)
        if not done11.startswith("DONE 11 "):
            print(f"FAIL: sessione pulita {done11!r}")
            return 1
        if reused_clean:
            print(f"FAIL: una sessione pulita dice di aver riusato {reused_clean} token")
            return 1
        fresh.stdin.close()
        fresh.wait(timeout=60)
    finally:
        if fresh.poll() is None:
            fresh.kill()

    if turn2 != clean:
        print(f"FAIL: il riuso del prefisso cambia la risposta\n"
              f"  con slot riusato: {turn2!r}\n"
              f"  da sessione pulita: {clean!r}")
        return 1

    # --- Continue senza la corsa di spazi finale ---
    #
    # Il gateway rifiuta un turno da continuare che finisce con spazi (il
    # template li toglie), e il client li toglie prima di mandarlo: il
    # Continue arriva corto della corsa di spazi che il motore ha generato e
    # gia' in cache. Il motore tiene lo stato di prima di quella corsa e
    # riprende da li'.
    #
    # Serve una risposta con testo e poi almeno due token di soli spazi. Il
    # modello a pesi casuali non si sceglie, quindi si cerca un prompt che la
    # dia. Il budget ferma il turno prima di macinare il suo ultimo token, e
    # si prova due volte:
    # - fino al secondo spazio compreso: il primo e' macinato, quindi la cache
    #   ha un token piu' del Continue tagliato e si deve davvero riavvolgere;
    # - un token dopo: sono macinati tutti e due, come dopo un client che se
    #   ne va, e il secondo passa anche lui dal punto dove si scatta. Lo
    #   scatto deve restare quello del primo. Un motore che scattasse a ogni
    #   spazio riprenderebbe solo dal secondo, il Continue non combacerebbe e
    #   si rifarebbe il prefill: giusto ma lento, e nessun altro caso se ne
    #   accorgerebbe. (Tre spazi di fila la fixture non li da' in 400 prompt.)
    spaces = set(b" \t\n\r\x0b\x0c")
    scout = engine(binary, arguments.fixture)
    found = None
    try:
        handshake(scout)
        for attempt in range(400):
            candidate = f"gu{attempt}".encode()
            submit_bytes(scout, 100 + attempt, candidate, max_tokens=48)
            pieces = []
            while True:
                line = read_line(scout.stdout)
                if line.startswith("DATA "):
                    count = int(line.split()[2])
                    pieces.append(scout.stdout.read(count + 1)[:count])
                elif line.startswith("DONE ") or line.startswith("ERROR "):
                    break
            for k in range(1, len(pieces) - 1):
                head = pieces[:k + 2]
                if all(len(piece) == 1 for piece in head) \
                        and head[k - 1][0] not in spaces \
                        and head[k][0] in spaces and head[k + 1][0] in spaces:
                    found = (candidate, pieces[:k], k)
                    break
            if found:
                break
        scout.stdin.close()
        scout.wait(timeout=60)
    finally:
        if scout.poll() is None:
            scout.kill()
    if not found:
        print("FAIL: nessun prompt fra 400 da' testo seguito da due spazi; la "
              "fixture e' cambiata e il caso del riavvolgimento non si prova")
        return 1
    blank_prompt, kept, k = found
    trimmed = blank_prompt + b"".join(kept)
    for fed, first in ((1, 40), (2, 44)):
        rewound = engine(binary, arguments.fixture, {"GLM53_REWIND": "1"})
        try:
            handshake(rewound)
            submit_bytes(rewound, first, blank_prompt, max_tokens=k + fed + 1)
            _, done_run, _ = collect(rewound, first)
            submit_bytes(rewound, first + 1, trimmed, max_tokens=4)
            from_rewind, done_continue, _ = collect(rewound, first + 1)
            rewound.stdin.close()
            rewound.wait(timeout=60)
        finally:
            if rewound.poll() is None:
                rewound.kill()
        if not done_run.startswith(f"DONE {first} ") \
                or not done_continue.startswith(f"DONE {first + 1} "):
            print(f"FAIL: riavvolgimento -> {done_run!r} / {done_continue!r}")
            return 1
        expected = [str(len(trimmed)), str(len(trimmed)), str(len(blank_prompt) + k + fed),
                    str(len(trimmed)), "shorter"]
        if reuse_line(first + 1)[2:] != expected:
            print(f"FAIL: il Continue senza i {fed} spazi finali in cache doveva "
                  f"riprendere dai {len(trimmed)} token prima della corsa: "
                  f"{reuse_line(first + 1)!r}")
            return 1
        if from_rewind != fresh_answer(binary, arguments.fixture, trimmed, 4):
            print(f"FAIL: riavvolto di {fed} spazi, il Continue risponde "
                  f"{from_rewind!r} e un motore appena partito no")
            return 1
    # Lo stesso senza GLM53_REWIND, cioe' il default: niente scatto, prefill
    # da capo.
    unwound = engine(binary, arguments.fixture)
    try:
        handshake(unwound)
        submit_bytes(unwound, 42, blank_prompt, max_tokens=k + 2)
        collect(unwound, 42)
        submit_bytes(unwound, 43, trimmed, max_tokens=4)
        collect(unwound, 43)
        unwound.stdin.close()
        unwound.wait(timeout=60)
    finally:
        if unwound.poll() is None:
            unwound.kill()
    if reuse_line(43)[2] != "0":
        print(f"FAIL: senza GLM53_REWIND il Continue non doveva riusare: "
              f"{reuse_line(43)!r}")
        return 1

    # --- CANCEL durante il prefill ---
    #
    # Il CANCEL si guarda anche fra un pezzo di prefill e l'altro: prima un
    # client che se ne andava a meta' di un prompt lungo lasciava il motore a
    # macinarlo fino in fondo per nessuno. Il motore si ferma a un confine di
    # pezzo, risponde CANCELLED senza DATA, e tiene quello che ha fatto: un
    # nuovo tentativo dello stesso prompt ne e' un'estensione stretta e lo
    # riusa, con la stessa risposta di una sessione pulita.
    #
    # Deterministico, non una corsa: il turno 29 mette in cache i primi 10
    # token, e il CANCEL del turno 30 parte nella stessa scrittura del SUBMIT,
    # quindi la guardata prima del primo pezzo lo trova gia' li'.
    long_prompt = prompt + "abcdefghijklmnopqrstuvwxyz0123"
    chunked = {"GLM53_PREFILL_CHUNK": "1"}
    halted = engine(binary, arguments.fixture, chunked)
    try:
        handshake(halted)
        submit(halted, 29, long_prompt[:10], max_tokens=1)
        _, done29, _ = collect(halted, 29)
        if not done29.startswith("DONE 29 "):
            print(f"FAIL: turno che prepara la cache -> {done29!r}")
            return 1
        body = long_prompt.encode()
        halted.stdin.write(f"SUBMIT 30 0 {len(body)} 4 0.0 1.0\n".encode() + body
                           + b"\nCANCEL 30\n")
        halted.stdin.flush()
        prefill_data, done30, _ = collect(halted, 30)
        if done30 != "ERROR 30 CANCELLED" or prefill_data:
            print(f"FAIL: CANCEL durante il prefill -> {done30!r} con "
                  f"{len(prefill_data)} byte di DATA, atteso ERROR 30 CANCELLED "
                  f"senza DATA")
            return 1
        notes = open(NOTES, "r", errors="replace").read().splitlines()
        halted_line = [line.split() for line in notes if line.startswith("CANCEL 30 ")]
        if halted_line != [["CANCEL", "30", str(len(body)), "0", "10"]]:
            print(f"FAIL: CANCEL durante il prefill doveva fermarsi ai 10 token "
                  f"gia' in cache su {len(body)}: {halted_line!r}")
            return 1
        submit(halted, 31, long_prompt, max_tokens=4)
        retried, done31, _ = collect(halted, 31)
        if not done31.startswith("DONE 31 "):
            print(f"FAIL: nuovo tentativo dopo il CANCEL -> {done31!r}")
            return 1
        if reuse_line(31)[2:4] != ["10", str(len(body))] or reuse_line(31)[-1] != "extend":
            print(f"FAIL: il nuovo tentativo doveva riusare i 10 token del prefill "
                  f"interrotto: {reuse_line(31)!r}")
            return 1
        halted.stdin.close()
        halted.wait(timeout=60)
    finally:
        if halted.poll() is None:
            halted.kill()
    clean_engine = engine(binary, arguments.fixture, chunked)
    try:
        handshake(clean_engine)
        # Lo stesso CANCEL su uno slot vuoto: si ferma prima di macinare
        # qualsiasi cosa, e la storia che resta ha zero token. Il motore deve
        # rispondere CANCELLED e poi servire il turno dopo da freddo.
        clean_engine.stdin.write(f"SUBMIT 33 0 {len(body)} 4 0.0 1.0\n".encode()
                                 + body + b"\nCANCEL 33\n")
        clean_engine.stdin.flush()
        _, done33, _ = collect(clean_engine, 33)
        empty_line = [line.split() for line in
                      open(NOTES, "r", errors="replace").read().splitlines()
                      if line.startswith("CANCEL 33 ")]
        if done33 != "ERROR 33 CANCELLED" or \
                empty_line != [["CANCEL", "33", str(len(body)), "0", "0"]]:
            print(f"FAIL: CANCEL prima del primo pezzo su uno slot vuoto -> "
                  f"{done33!r}, {empty_line!r}")
            return 1
        submit(clean_engine, 32, long_prompt, max_tokens=4)
        from_clean, done32, _ = collect(clean_engine, 32)
        clean_engine.stdin.close()
        clean_engine.wait(timeout=60)
    finally:
        if clean_engine.poll() is None:
            clean_engine.kill()
    if reuse_line(32)[-1:] != ["cold"]:
        print(f"FAIL: dopo un CANCEL a zero token il turno doveva partire da "
              f"freddo: {reuse_line(32)!r}")
        return 1
    if not done32.startswith("DONE 32 ") or retried != from_clean:
        print(f"FAIL: dopo un prefill interrotto la risposta cambia\n"
              f"  riusando: {retried!r}\n  da pulito: {from_clean!r} ({done32!r})")
        return 1

    # --- CANCEL a meta' di un pezzo di prefill ---
    #
    # Fra un pezzo e l'altro non basta: un pezzo puo' durare a lungo. Il
    # CANCEL si guarda anche fra un layer e l'altro, e un pezzo fermato a
    # meta' si butta rimettendo lo stato KDA di inizio pezzo. Qui un solo
    # pezzo lungo (1000 token, 4 layer) e un CANCEL
    # mandato dopo un po': deve fermarlo dentro il pezzo (riga HALT), lasciare
    # la cache ai 10 token di prima, e il nuovo tentativo deve riusarli e
    # rispondere come un motore appena partito. Se lo stato non venisse
    # rimesso, il riuso partirebbe da una ricorrenza avanzata a meta'.
    #
    # E' l'unico caso a tempo: se il CANCEL arriva prima o dopo il pezzo si
    # riprova con un altro ritardo, e senza una riga HALT il caso fallisce
    # invece di passare senza aver provato niente. I ritardi sono frazioni di
    # quanto dura il pezzo su questa macchina, misurato prima senza CANCEL:
    # ritardi fissi andavano bene su una macchina e su un'altra, dieci volte
    # piu' veloce, arrivavano tutti dopo il pezzo.
    one_chunk = {"GLM53_PREFILL_CHUNK": "4096"}
    wide = ("gu" + "abcdefghijklmnopqrstuvwxyz0123" * 40)[:1000].encode()
    timing = engine(binary, arguments.fixture, one_chunk)
    try:
        handshake(timing)
        submit_bytes(timing, 50, wide[:10], max_tokens=1)
        collect(timing, 50)
        started = time.monotonic()
        submit_bytes(timing, 51, wide, max_tokens=1)
        collect(timing, 51)
        took = time.monotonic() - started
    finally:
        timing.stdin.close()
        timing.wait(timeout=120)
    delays = [took * share for share in (0.4, 0.25, 0.55, 0.15, 0.7)]
    halted_mid = None
    for delay in delays:
        mid = engine(binary, arguments.fixture, one_chunk)
        try:
            handshake(mid)
            submit_bytes(mid, 50, wide[:10], max_tokens=1)
            collect(mid, 50)
            submit_bytes(mid, 51, wide, max_tokens=4)
            time.sleep(delay)
            mid.stdin.write(b"CANCEL 51\n")
            mid.stdin.flush()
            _, done51, _ = collect(mid, 51)
            notes = open(NOTES, "r", errors="replace").read().splitlines()
            if not any(line.startswith("HALT ") for line in notes):
                continue                      # fuori dal pezzo: altro ritardo
            submit_bytes(mid, 52, wide, max_tokens=4)
            retried_mid, done52, _ = collect(mid, 52)
            halted_mid = (done51, [line.split() for line in notes
                                   if line.startswith("CANCEL 51 ")],
                          retried_mid, done52, reuse_line(52))
            break
        finally:
            mid.stdin.close()
            mid.wait(timeout=120)
    if not halted_mid:
        print("FAIL: nessun CANCEL e' arrivato dentro il pezzo di prefill con "
              f"nessuno dei ritardi provati (nessuna riga HALT): il pezzo dura "
              f"{took:.3f} s, ritardi {', '.join(f'{d:.3f}' for d in delays)} s")
        return 1
    done51, cancel51, retried_mid, done52, reuse52 = halted_mid
    if done51 != "ERROR 51 CANCELLED" or \
            cancel51 != [["CANCEL", "51", str(len(wide)), "0", "10"]]:
        print(f"FAIL: CANCEL dentro il pezzo -> {done51!r}, {cancel51!r}; attesi "
              f"CANCELLED e la cache ferma ai 10 token di prima")
        return 1
    if not done52.startswith("DONE 52 ") or reuse52[2:4] != ["10", str(len(wide))] \
            or reuse52[-1] != "extend":
        print(f"FAIL: dopo un pezzo buttato il nuovo tentativo doveva riusare i 10 "
              f"token: {done52!r}, {reuse52!r}")
        return 1
    if retried_mid != fresh_answer(binary, arguments.fixture, wide, 4, one_chunk):
        print(f"FAIL: dopo un pezzo buttato la risposta cambia ({retried_mid!r}): "
              f"lo stato KDA non e' tornato all'inizio del pezzo")
        return 1

    # La CLI stampa la risposta e poi un a capo; quello che conta e' che i byte
    # della risposta siano gli stessi.
    from_cli = cli_answer(binary, arguments.fixture, prompt, tokens)
    if served not in from_cli:
        print(f"FAIL: il server e la CLI non rispondono uguale\n"
              f"  server: {served!r}\n  CLI:    {from_cli!r}")
        return 1

    print(f"PASS GLM-5.3 serve: handshake, frame a byte contati, {emitted} token "
          f"decodificati identici alla CLI, prompt vuoto rifiutato, stream "
          f"ancora allineato dopo l'errore, {reused} token di prefisso riusati "
          f"al secondo turno con la stessa risposta di una sessione pulita, "
          f"CANCEL onorato a meta' turno dopo {emitted_before} token su {budget}, "
          f"STOP chiuso col DONE dopo {emitted_stop} token su {stop_budget}, "
          f"Continue senza gli spazi finali ripreso a {len(trimmed)} token, "
          f"CANCEL dentro un pezzo di prefill buttato e rimesso a 10 token, "
          f"CANCEL durante il prefill fermo a 10 token su {len(body)} e riusato "
          f"dal nuovo tentativo, "
          f"SUBMIT a slot occupato rifiutato con SLOT_BUSY, "
          f"{emitted_eof} token portati a termine con la pipa gia' chiusa")
    return 0


if __name__ == "__main__":
    sys.exit(main())
