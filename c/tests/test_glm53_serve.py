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
        return 0
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
        while True:
            line = read_line(process.stdout)
            if not line.startswith("DATA "):
                break
            _, got_id, count = line.split()
            if int(got_id) != 12:
                raise AssertionError(f"DATA per {got_id}, atteso 12")
            process.stdout.read(int(count) + 1)
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
          f"SUBMIT a slot occupato rifiutato con SLOT_BUSY, "
          f"{emitted_eof} token portati a termine con la pipa gia' chiusa")
    return 0


if __name__ == "__main__":
    sys.exit(main())
