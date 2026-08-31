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
 * strtol invece dice dove ha smesso di leggere, e qui si pretende che abbia
 * letto tutto. */
#ifndef COLI_CLI_ARGS_H
#define COLI_CLI_ARGS_H

#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>

/* `what` nomina l'argomento come lo chiama chi lo scrive sulla riga di comando
 * ("cache/layer", non "cap"): il messaggio serve a chi ha sbagliato a digitare,
 * non a chi legge il sorgente. */
static int coli_arg_int(const char *arg, const char *what)
{
    char *end = NULL;
    long v;

    errno = 0;
    v = strtol(arg, &end, 10);
    /* Prima cosa: end == arg vuol dire che non ha letto nemmeno una cifra.
     * Va controllato PRIMA di saltare gli spazi finali, se no un argomento
     * fatto di soli spazi passerebbe valendo zero -- lo skip sposterebbe end
     * oltre lo spazio e la stringa sembrerebbe consumata per intero. */
    if (end == arg) goto bad;
    /* strtol salta gli spazi iniziali da solo; quelli finali li salto qui, se
     * no un "32\r" arrivato da uno script salvato con le terminazioni di riga
     * di Windows verrebbe rifiutato con un messaggio incomprensibile, e i
     * binari Windows li spediamo. */
    while (*end == ' ' || *end == '\t' || *end == '\r' || *end == '\n') end++;
    /* *end != 0: ne ha letta una parte e poi ha trovato altro, il caso "3x2". */
    if (*end != '\0' || errno == ERANGE ||
        v < (long)INT_MIN || v > (long)INT_MAX) {
bad:
        fprintf(stderr, "%s: expected a whole number, got \"%s\"\n", what, arg);
        exit(2);
    }
    return (int)v;
}

#endif /* COLI_CLI_ARGS_H */
