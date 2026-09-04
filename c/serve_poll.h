/* serve_poll.h — guardare stdin SENZA bloccarsi, per onorare un CANCEL mentre
 * il turno e' ancora in corso.
 *
 * #1332: il protocollo del gateway documenta "CANCEL <id> (abort current turn)",
 * ma nella maggior parte dei motori il ciclo di decode non legge stdin mentre
 * gira: il comando viene visto solo FRA una richiesta e l'altra, cioe' quando
 * non serve piu'. Il gateway intanto manda CANCEL e aspetta l'ack tenendo
 * l'ammissione dello scheduler, quindi un client che si disconnette non libera
 * niente: si aspetta comunque la fine del turno.
 *
 * Due motori lo facevano gia', ciascuno per conto suo: colibri (#678,
 * mux_ctl_poll) e kimi_k3. Questo header estrae quella primitiva -- la stessa,
 * gia' provata sul campo -- perche' gli altri non debbano reinventarla.
 *
 * Cosa fa: dice se c'e' almeno un byte leggibile su stdin, subito e senza
 * attendere. Cosa NON fa: leggere. Il formato delle righe lo conosce il
 * chiamante (serve_codec.h per cinque motori, un parser proprio per gli altri),
 * e mescolare le due cose qui vorrebbe dire riscrivere il codec.
 *
 * Regola sui casi che non sappiamo gestire: se la piattaforma non sa dire se
 * stdin e' pronto, si risponde "non pronto" e il comportamento torna quello di
 * prima -- il CANCEL a fine turno. Mai indovinare "pronto": una lettura
 * bloccante dentro il ciclo di decode fermerebbe la generazione per sempre in
 * attesa di un comando che potrebbe non arrivare mai. */
#ifndef COLI_SERVE_POLL_H
#define COLI_SERVE_POLL_H

#if defined(_WIN32)
#include <io.h>
#include <windows.h>
#else
#include <sys/select.h>
#include <unistd.h>
#endif

#ifndef STDIN_FILENO
#define STDIN_FILENO 0
#endif

/* 1 = c'e' qualcosa da leggere su stdin adesso; 0 = niente, o piattaforma che
 * non sa dirlo. Non blocca mai. */
static inline int coli_serve_stdin_ready(void)
{
#if defined(_WIN32)
    HANDLE handle = (HANDLE)_get_osfhandle(_fileno(stdin));
    DWORD available = 0;
    /* PeekNamedPipe funziona sulle pipe, che e' come il gateway parla ai
     * motori. Su una console o un file ritorna falso e si resta al
     * comportamento di prima: nessun falso "pronto". */
    if (PeekNamedPipe(handle, NULL, 0, NULL, &available, NULL) && available > 0)
        return 1;
    return 0;
#elif defined(__linux__) || defined(__APPLE__) || defined(__FreeBSD__)
    fd_set readable;
    struct timeval immediately = {0, 0};
    FD_ZERO(&readable);
    FD_SET(STDIN_FILENO, &readable);
    return select(STDIN_FILENO + 1, &readable, NULL, NULL, &immediately) > 0
        && FD_ISSET(STDIN_FILENO, &readable);
#else
    return 0;   /* piattaforma sconosciuta: mai fingere che ci sia input */
#endif
}

#endif /* COLI_SERVE_POLL_H */
