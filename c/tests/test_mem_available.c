/* La memoria disponibile deve essere MISURATA su ogni piattaforma che
 * spediamo, non letta da un file che esiste solo su Linux.
 *
 * #1375: glm53.c leggeva /proc/meminfo ovunque. Su Windows il file non c'e',
 * la funzione tornava 0, il budget della cache esperti si clampava a 1 GB e
 * GLM-5.3-Flash girava con uno slot per layer -- in silenzio, per mesi,
 * perche' nessun test chiedeva alla misura di essere maggiore di zero.
 *
 * Questo test gira nei job Linux, macOS e Windows della CI e pretende tre
 * cose dalla funzione condivisa in compat.h: che misuri qualcosa, che il
 * numero stia dentro la RAM fisica, e -- dove il sistema espone il proprio
 * MemAvailable -- che coincida con quello. Il terzo vincolo e' quello che
 * impedisce di "passare" restituendo una costante. */
#include <stdio.h>
#include <stdlib.h>
#include "../compat.h"

static int fails;
static void check(int ok, const char *what){ if(!ok){ printf("  FAIL: %s\n", what); fails++; } }

/* Il totale arriva ora dalla stessa funzione condivisa della disponibile
 * (compat_meminfo_gb), su tutte e tre le piattaforme. Prima questo test
 * tornava 0 su macOS -- "nessun totale a portata senza sysctl" -- e il
 * vincolo "disponibile <= RAM fisica" saltava proprio sulla piattaforma dove
 * la disponibile e' una stima. Ora non salta piu' da nessuna parte. */
static double total_gb(void){
    double t = 0, a = 0;
    compat_meminfo_gb(&t, &a);
    return t;
}

int main(void){
    double avail = compat_mem_available_gb();
    /* compat_mem_available_gb() e' ora un wrapper su compat_meminfo_gb(): le
     * due devono dare lo stesso numero, o il wrapper ha perso qualcosa. Sono
     * pero' due letture del sistema a qualche microsecondo di distanza, e la
     * memoria disponibile si muove fra l'una e l'altra (il runner Windows della
     * CI ha fallito il confronto esatto sul PR #1656 senza toccare questo
     * codice): un quarto di GB di tolleranza distingue "il wrapper ha perso
     * qualcosa" (differenze di GB) dal rumore di due istanti diversi. */
    { double t2 = 0, a2 = 0; compat_meminfo_gb(&t2, &a2);
      double gap = a2 > avail ? a2 - avail : avail - a2;
      check(gap < 0.25, "compat_meminfo_gb e compat_mem_available_gb non concordano"); }
    printf("  disponibile: %.2f GB\n", avail);
    check(avail > 0.0, "la misura vale 0: la piattaforma non e' coperta (era il bug di Windows)");
    double total = total_gb();
    if(total > 0){
        printf("  totale:      %.2f GB\n", total);
        check(avail <= total * 1.001, "disponibile > RAM fisica: la misura non e' una misura");
    }
#if !defined(_WIN32) && !defined(__APPLE__)
    /* Linux: deve essere MemAvailable, non un'altra riga di meminfo. */
    FILE *f = fopen("/proc/meminfo", "r"); double kb = 0;
    if(f){ char ln[256]; while(fgets(ln, sizeof ln, f)) if(sscanf(ln, "MemAvailable: %lf", &kb) == 1) break; fclose(f); }
    if(kb > 0){
        double ratio = avail / (kb / 1e6);
        check(ratio > 0.9 && ratio < 1.1, "non coincide con MemAvailable di /proc/meminfo");
    }
#endif
    if(fails){ printf("test_mem_available: %d fallimenti\n", fails); return 1; }
    printf("test_mem_available: ok\n"); return 0;
}
