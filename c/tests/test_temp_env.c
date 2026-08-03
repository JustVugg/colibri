/* Regressione #509: il canale della temperatura era l'env var TEMP, che ROCm
 * (comgr/MIOpen) e Windows possiedono gia' come path della directory temporanea.
 * Due danni: (a) `coli --temp 0.6` esportava TEMP=0.6 e l'init HIP moriva dentro
 * comgr (SIGSEGV su gfx1100, hipErrorOutOfMemory su gfx1030); (b) sulle build
 * native Windows TEMP e' SEMPRE definita come path e atof("C:\...\Temp")==0
 * forzava silenziosamente greedy su ogni run senza override.
 *
 * Contratto di temp_from_env(COLI_TEMP, TEMP):
 *   - COLI_TEMP presente -> vince sempre (canale primario);
 *   - altrimenti TEMP e' accettata SOLO se interamente numerica (alias legacy);
 *   - un TEMP non numerico (path) o assente -> -1 = auto. */
#define main coli_glm_main_unused
#include "../colibri.c"
#undef main
#include <stdio.h>

static int feq(float a, float b) { return fabsf(a - b) < 1e-6f; }

int main(void) {
    int fail = 0;
    struct {
        const char *ct, *t;
        float want;
        const char *why;
    } C[] = {
        {"0.6", NULL, 0.6f, "COLI_TEMP da solo"},
        {"0", "/tmp", 0.0f, "COLI_TEMP=0 (greedy) vince su un TEMP-directory"},
        {"0.7", "0.2", 0.7f, "COLI_TEMP batte un TEMP numerico legacy"},
        {NULL, "0.6", 0.6f, "alias legacy: TEMP interamente numerico"},
        {NULL, "0", 0.0f, "alias legacy: TEMP=0 = greedy (test interni)"},
        {NULL, "C:\\Users\\u\\AppData\\Local\\Temp", -1.f, "Windows: il path NON e' una temperatura"},
        {NULL, "/tmp", -1.f, "POSIX: una directory NON e' una temperatura"},
        {NULL, "0.6abc", -1.f, "numero+spazzatura: rifiutato per intero"},
        {NULL, "", -1.f, "stringa vuota -> auto"},
        {NULL, NULL, -1.f, "niente env -> auto"},
        {"-1", "C:\\Temp", -1.f, "COLI_TEMP=-1 esplicito = auto"},
    };
    int n = (int)(sizeof C / sizeof C[0]);
    for (int i = 0; i < n; i++) {
        float got = temp_from_env(C[i].ct, C[i].t);
        if (!feq(got, C[i].want)) {
            printf("  FAIL: (COLI_TEMP=%s, TEMP=%s) -> %g, atteso %g  [%s]\n", C[i].ct ? C[i].ct : "(unset)",
                   C[i].t ? C[i].t : "(unset)", got, C[i].want, C[i].why);
            fail = 1;
        }
    }
    if (!fail) printf("  %d casi: primario/alias/path-Windows/auto              ok\n", n);
    printf(fail ? "test_temp_env: FAIL\n" : "test_temp_env: ok\n");
    return fail;
}
