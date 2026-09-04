/* coli_serve_stdin_ready: la guardata a stdin che rende possibile un CANCEL a
 * meta' turno (#1332).
 *
 * La proprieta' pericolosa e' una sola, e questo test esiste per quella: la
 * funzione NON DEVE MAI BLOCCARE. Viene chiamata dentro il ciclo di decode, una
 * volta per token; se un giorno qualcuno la trasformasse in una lettura
 * bloccante, la generazione si fermerebbe ad aspettare un comando che potrebbe
 * non arrivare mai -- un guasto peggiore di quello che la funzione cura.
 *
 * Percio' qui il tempo si misura davvero invece di fidarsi: con stdin vuoto la
 * chiamata deve tornare entro pochi millisecondi. Un test che guardasse solo il
 * valore di ritorno passerebbe anche con una versione bloccante, purche' prima
 * o poi qualcuno scrivesse qualcosa. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "../serve_poll.h"

static int fails;
static void check(int ok, const char *what) {
    if (!ok) { printf("  FAIL: %s\n", what); fails++; }
}

static double seconds_since(struct timespec start) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (now.tv_sec - start.tv_sec) + (now.tv_nsec - start.tv_nsec) / 1e9;
}

int main(void) {
    int pipefd[2];
    if (pipe(pipefd) != 0) { perror("pipe"); return 1; }

    /* stdin diventa il lato di lettura della pipe: e' come il gateway parla ai
     * motori, e insieme il caso che PeekNamedPipe gestisce su Windows. */
    if (dup2(pipefd[0], STDIN_FILENO) < 0) { perror("dup2"); return 1; }

    /* ---- 1. pipe vuota: niente da leggere, e SUBITO ---- */
    struct timespec t0;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    int ready = coli_serve_stdin_ready();
    double elapsed = seconds_since(t0);
    check(ready == 0, "pipe vuota: non deve esserci niente da leggere");
    check(elapsed < 0.25,
          "la chiamata ha impiegato troppo: se blocca, il ciclo di decode si "
          "ferma ad aspettare un comando che puo' non arrivare mai");

    /* ---- 2. arriva un CANCEL: ora c'e' qualcosa ---- */
    const char *command = "CANCEL r1\n";
    if (write(pipefd[1], command, strlen(command)) < 0) { perror("write"); return 1; }
    clock_gettime(CLOCK_MONOTONIC, &t0);
    ready = coli_serve_stdin_ready();
    elapsed = seconds_since(t0);
    check(ready == 1, "con un comando in coda deve dire che stdin e' pronto");
    check(elapsed < 0.25, "anche con dati pronti la chiamata deve tornare subito");

    /* ---- 3. la riga si legge davvero, e dopo la pipe torna vuota ---- */
    char line[128] = {0};
    if (!fgets(line, sizeof(line), stdin)) { printf("  FAIL: fgets non legge\n"); return 1; }
    check(strncmp(line, "CANCEL r1", 9) == 0, "la riga letta non e' quella scritta");
    clock_gettime(CLOCK_MONOTONIC, &t0);
    check(coli_serve_stdin_ready() == 0, "dopo il consumo la pipe deve tornare vuota");
    check(seconds_since(t0) < 0.25, "la chiamata su pipe svuotata deve tornare subito");

    /* ---- 4. chiusura in scrittura: pronto (una lettura darebbe EOF) ---- */
    close(pipefd[1]);
    check(coli_serve_stdin_ready() == 1,
          "una pipe chiusa e' 'pronta': la lettura ritorna EOF, e il chiamante "
          "deve poterlo vedere invece di restare appeso");

    if (fails) { printf("test_serve_poll: %d fallimenti\n", fails); return 1; }
    printf("test_serve_poll: ok\n");
    return 0;
}
