/* coli_parse_int: cio' che atoi accettava a meta' ora e' un rifiuto.
 *
 * Il caso che ha motivato tutto questo e' "3x2" scritto al posto di "32": atoi
 * restituiva 3, il motore partiva con una cache dieci volte piu' piccola, e
 * l'unico sintomo era che andava piano. Nessun messaggio.
 *
 * Si prova il parser puro e non il guscio che esce: cosi' il test e' un
 * confronto fra valori, senza fork() -- che su Windows non c'e'. */
#include <stdio.h>
#include <string.h>

#include "../cli_args.h"

static int fails = 0;

static void accepts(const char *arg, int expect)
{
    int got = -12345;
    if (!coli_parse_int(arg, &got)) {
        printf("  FAIL \"%s\": rifiutato, doveva valere %d\n", arg, expect);
        fails++;
    } else if (got != expect) {
        printf("  FAIL \"%s\": vale %d invece di %d\n", arg, got, expect);
        fails++;
    }
}

static void rejects(const char *arg)
{
    int got = -12345;
    if (coli_parse_int(arg, &got)) {
        printf("  FAIL \"%s\": accettato come %d -- e' il bug originale, un "
               "argomento non valido letto a meta'\n", arg, got);
        fails++;
    } else if (got != -12345) {
        printf("  FAIL \"%s\": rifiutato ma ha scritto %d nell'uscita\n",
               arg, got);
        fails++;
    }
}

int main(void)
{
    /* Quello che deve continuare a funzionare. */
    accepts("32", 32);
    accepts("0", 0);          /* il limite cap >= 1 e' del chiamante, non di qui */
    accepts("-5", -5);
    accepts("+32", 32);
    accepts(" 32", 32);       /* strtol salta gli spazi iniziali */
    accepts("32 ", 32);       /* e noi quelli finali */
    accepts("32\r", 32);      /* uno script salvato con le terminazioni Windows */
    accepts("32\n", 32);
    accepts("2147483647", 2147483647);

    /* Quello che prima passava in silenzio. */
    rejects("3x2");           /* il refuso vero: atoi diceva 3 */
    rejects("32abc");
    rejects("abc");
    rejects("");
    rejects(" ");
    rejects("0x20");          /* base 16 non e' quello che intendeva chi scrive */
    rejects("3.5");
    rejects("99999999999999999999");   /* ERANGE */
    rejects("-99999999999999999999");
    rejects("2147483648");    /* un piu' di INT_MAX: non entra in un int */
    rejects(NULL);

    if (fails) { printf("test_cli_args: %d fallimenti\n", fails); return 1; }
    printf("test_cli_args: ok\n");
    return 0;
}
