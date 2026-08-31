/* coli_arg_int: cio' che atoi accettava a meta' ora e' un errore.
 *
 * Il caso che ha motivato tutto questo e' "3x2" scritto al posto di "32": atoi
 * restituiva 3, il motore partiva con una cache dieci volte piu' piccola, e
 * l'unico sintomo era che andava piano. Nessun messaggio.
 *
 * coli_arg_int chiama exit(2) quando rifiuta, quindi ogni caso si prova in un
 * processo figlio: e' l'uscita che interessa, non un valore di ritorno. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

#include "../cli_args.h"

static int fails = 0;

/* Ritorna il codice di uscita del figlio, o -1 se e' morto di segnale. */
static int run_in_child(const char *arg, int *value_out)
{
    int fd[2], status;
    pid_t pid;
    int v = 0;

    if (pipe(fd) != 0) return -1;
    /* Senza questo il figlio eredita cio' che il padre ha ancora nel buffer di
     * stdout, e quando coli_arg_int chiama exit() lo scarica: ogni FAIL gia'
     * stampato ricompare una volta per ogni caso successivo. */
    fflush(NULL);
    pid = fork();
    if (pid == 0) {
        close(fd[0]);
        v = coli_arg_int(arg, "cache/layer");
        /* Ci arriva solo se ha accettato. */
        if (write(fd[1], &v, sizeof v) != (ssize_t)sizeof v) _exit(3);
        _exit(0);
    }
    close(fd[1]);
    if (read(fd[0], &v, sizeof v) == (ssize_t)sizeof v && value_out) *value_out = v;
    close(fd[0]);
    waitpid(pid, &status, 0);
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

static void accepts(const char *arg, int expect)
{
    int got = 0;
    int code = run_in_child(arg, &got);
    if (code != 0) {
        printf("  FAIL \"%s\": rifiutato (exit %d), doveva valere %d\n",
               arg, code, expect);
        fails++;
    } else if (got != expect) {
        printf("  FAIL \"%s\": vale %d invece di %d\n", arg, got, expect);
        fails++;
    }
}

static void rejects(const char *arg)
{
    int code = run_in_child(arg, NULL);
    if (code != 2) {
        printf("  FAIL \"%s\": accettato o uscita %d invece di 2 -- e' il bug "
               "originale, un argomento non valido letto a meta'\n", arg, code);
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

    if (fails) { printf("test_cli_args: %d fallimenti\n", fails); return 1; }
    printf("test_cli_args: ok\n");
    return 0;
}
