/* Lettura degli argomenti numerici dalla riga di comando.
 *
 * atoi() non ha modo di dire "non e' un numero": restituisce 0 per una stringa
 * che non lo e', e per una che lo e' a meta' restituisce la meta' che ha letto
 * senza dire niente. Il caso che si vede davvero e' un refuso nella capienza:
 *
 *     ./qwen38 3x2      ->  atoi = 3     ->  cache=3/layer invece di 32
 *
 * Nessun errore. Il motore parte con una cache dieci volte piu' piccola, legge
 * dal disco molto piu' del dovuto, e chi l'ha lanciato conclude che il motore
 * e' lento invece che di aver sbagliato a digitare. E' il modo peggiore in cui
 * un programma puo' sbagliare: fa una cosa diversa da quella chiesta e non lo
 * dice.
 *
 * Due funzioni e non una: coli_parse_int decide e riporta, coli_arg_int e' il
 * guscio che stampa ed esce. La divisione non e' estetica, e' quello che rende
 * la decisione verificabile senza un processo figlio -- e quindi su Windows,
 * che non ha fork(). */
#ifndef COLI_CLI_ARGS_H
#define COLI_CLI_ARGS_H

#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>

/* 1 se `arg` e' un intero scritto per intero, 0 altrimenti. *out tocco solo
 * quando accetto. */
static int coli_parse_int(const char *arg, int *out)
{
    char *end = NULL;
    long v;

    if (!arg) return 0;
    errno = 0;
    v = strtol(arg, &end, 10);
    /* Prima cosa: end == arg vuol dire che non ha letto nemmeno una cifra.
     * Va controllato PRIMA di saltare gli spazi finali, se no un argomento
     * fatto di soli spazi passerebbe valendo zero -- lo skip sposterebbe end
     * oltre lo spazio e la stringa sembrerebbe consumata per intero. */
    if (end == arg) return 0;
    /* strtol salta gli spazi iniziali da solo; quelli finali li salto qui, se
     * no un "32\r" arrivato da uno script salvato con le terminazioni di riga
     * di Windows verrebbe rifiutato con un messaggio incomprensibile, e i
     * binari Windows li spediamo. */
    while (*end == ' ' || *end == '\t' || *end == '\r' || *end == '\n') end++;
    /* *end != 0: ne ha letta una parte e poi ha trovato altro, il caso "3x2". */
    if (*end != '\0' || errno == ERANGE ||
        v < (long)INT_MIN || v > (long)INT_MAX) return 0;
    *out = (int)v;
    return 1;
}

/* `what` nomina l'argomento come lo chiama chi lo scrive sulla riga di comando
 * ("cache/layer", non "cap"): il messaggio serve a chi ha sbagliato a digitare,
 * non a chi legge il sorgente. */
static int coli_arg_int(const char *arg, const char *what)
{
    int v = 0;
    if (!coli_parse_int(arg, &v)) {
        fprintf(stderr, "%s: expected a whole number, got \"%s\"\n",
                what, arg ? arg : "");
        exit(2);
    }
    return v;
}

#endif /* COLI_CLI_ARGS_H */
