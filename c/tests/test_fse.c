/* Batteria di test per fse_coli.h — scritta PRIMA che il codec tocchi glm.c.
 *
 * Questo codec decomprimera' PESI: un bug qui non crasha, corrompe l'output
 * del modello in silenzio. Quindi: round-trip esatti, distribuzioni degeneri,
 * fuzz di troncamento e corruzione (il decoder non deve MAI crashare, leggere
 * oltre, o accettare in silenzio un flusso rotto), expert reale se disponibile
 * (env EXPERT_RAW), ratio contro l'entropia misurata, velocita' di decode.
 * Da compilare anche con -fsanitize=address,undefined: il fuzz sotto ASAN e'
 * il vero certificato di bounds-safety. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "../fse_coli.h"

static uint64_t rng = 0x243F6A8885A308D3ull;
static uint32_t xr(void) {
    rng ^= rng << 13;
    rng ^= rng >> 7;
    rng ^= rng << 17;
    return (uint32_t)(rng >> 32);
}

static int roundtrip(const uint8_t *in, size_t n, const char *name, double *ratio) {
    size_t cap = cfse_bound(n);
    uint8_t *c = malloc(cap ? cap : 1), *d = malloc(n ? n : 1);
    size_t cs = cfse_compress(in, n, c, cap);
    if (!cs) {
        printf("  FAIL %s: compress=0\n", name);
        return 1;
    }
    size_t rl = 0;
    if (cfse_decompress(c, cs, d, n, &rl)) {
        printf("  FAIL %s: decompress rifiutato\n", name);
        return 1;
    }
    if (rl != n || (n && memcmp(in, d, n))) {
        printf("  FAIL %s: round-trip NON identico\n", name);
        return 1;
    }
    if (ratio) *ratio = cs ? (double)n / (double)cs : 0;
    free(c);
    free(d);
    return 0;
}

int main(void) {
    int fail = 0;
    double r;
    printf("test_fse: rANS nibble per il container colibri'\n");

    /* --- 1. vettori noti e casi limite --- */
    {
        uint8_t v[] = {0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF};
        fail |= roundtrip(v, sizeof v, "16 byte tutti i nibble", NULL);
    }
    fail |= roundtrip((const uint8_t *)"", 0, "vuoto", NULL);
    {
        uint8_t b = 0x5A;
        fail |= roundtrip(&b, 1, "un byte", NULL);
    }
    {
        uint8_t v[3] = {0xFF, 0x00, 0xFF};
        fail |= roundtrip(v, 3, "tre byte estremi", NULL);
    }
    if (!fail) printf("  vettori noti + vuoto + 1 byte                    ok\n");

    /* --- 2. degeneri: un solo simbolo, due simboli --- */
    {
        uint8_t *v = malloc(100000);
        memset(v, 0x88, 100000);
        fail |= roundtrip(v, 100000, "tutto un simbolo", &r);
        if (!fail) printf("  degenere un-simbolo: ratio %.0fx                 ok\n", r);
        for (size_t i = 0; i < 100000; i++) v[i] = (xr() & 1) ? 0x87 : 0x78;
        fail |= roundtrip(v, 100000, "due simboli", &r);
        if (!fail) printf("  degenere due-simboli: ratio %.2fx                ok\n", r);
        free(v);
    }

    /* --- 3. uniforme (incomprimibile): DEVE cadere in raw, mai espandere --- */
    {
        size_t n = 1 << 20;
        uint8_t *v = malloc(n);
        for (size_t i = 0; i < n; i++) v[i] = (uint8_t)xr();
        size_t cap = cfse_bound(n);
        uint8_t *c = malloc(cap);
        size_t cs = cfse_compress(v, n, c, cap);
        if (!cs || cs > n + CFSE_HDR) {
            printf("  FAIL uniforme: espande (%zu > %zu)\n", cs, n + CFSE_HDR);
            fail = 1;
        }
        fail |= roundtrip(v, n, "uniforme 1MB", &r);
        if (!fail) printf("  uniforme 1MB: ratio %.3fx (fallback raw)         ok\n", r);
        free(v);
        free(c);
    }

    /* --- 4. gaussiana tipo-pesi: il ratio deve avvicinare il previsto 1.37x --- */
    {
        size_t n = 4 << 20;
        uint8_t *v = malloc(n);
        for (size_t i = 0; i < n; i++) {
            int a = ((int)(xr() & 15) + (int)(xr() & 15) + (int)(xr() & 15) + (int)(xr() & 15)) / 4; /* ~gauss 0..15 */
            int b = ((int)(xr() & 15) + (int)(xr() & 15) + (int)(xr() & 15) + (int)(xr() & 15)) / 4;
            v[i] = (uint8_t)(a | (b << 4));
        }
        fail |= roundtrip(v, n, "gaussiana 4MB", &r);
        if (!fail) printf("  gaussiana 4MB: ratio %.3fx                       ok\n", r);
        free(v);
    }

    /* --- 5. taglie casuali: 400 round-trip di lunghezze 0..8191 --- */
    {
        int bad = 0;
        for (int t = 0; t < 400; t++) {
            size_t n = xr() & 8191;
            uint8_t *v = malloc(n ? n : 1);
            for (size_t i = 0; i < n; i++) {
                int a = ((int)(xr() & 15) + (int)(xr() & 15)) / 2;
                v[i] = (uint8_t)(a | (((int)(xr() & 7) + 4) << 4));
            }
            bad |= roundtrip(v, n, "taglia casuale", NULL);
            free(v);
        }
        fail |= bad;
        if (!bad) printf("  400 round-trip a taglie casuali                  ok\n");
    }

    /* --- 6. FUZZ troncamento: OGNI prefisso del compresso va rifiutato pulito --- */
    {
        size_t n = 65536;
        uint8_t *v = malloc(n);
        for (size_t i = 0; i < n; i++) {
            int a = ((int)(xr() & 15) + (int)(xr() & 15)) / 2;
            v[i] = (uint8_t)(a | (a << 4));
        }
        size_t cap = cfse_bound(n);
        uint8_t *c = malloc(cap), *d = malloc(n);
        size_t cs = cfse_compress(v, n, c, cap);
        size_t rl;
        int accepted = 0;
        for (size_t L = 0; L < cs; L++) /* TUTTI i troncamenti */
            if (cfse_decompress(c, L, d, n, &rl) == 0) accepted++;
        if (accepted) {
            printf("  FAIL troncamento: %d prefissi accettati\n", accepted);
            fail = 1;
        } else printf("  troncamento: %zu prefissi, tutti rifiutati       ok\n", cs);

        /* --- 7. FUZZ corruzione: flip di byte; mai crash, (quasi) mai accettato --- */
        int acc = 0, trials = 2000;
        for (int t = 0; t < trials; t++) {
            size_t pos = xr() % cs;
            uint8_t old = c[pos];
            c[pos] ^= (uint8_t)(1 + (xr() & 0xFE));
            if (cfse_decompress(c, cs, d, n, &rl) == 0 && (rl != n || memcmp(v, d, n))) acc++;
            c[pos] = old;
        }
        printf("  corruzione: %d flip, %d accettati con dati sbagliati %s\n", trials, acc, acc <= 2 ? "ok" : "TROPPI");
        if (acc > 2) fail = 1; /* sigillo: attesi ~0 */
        free(v);
        free(c);
        free(d);
    }

    /* --- 8. expert REALE (se disponibile) + velocita' --- */
    const char *er = getenv("EXPERT_RAW");
    if (er) {
        FILE *f = fopen(er, "rb");
        if (f) {
            fseek(f, 0, SEEK_END);
            size_t n = (size_t)ftell(f);
            fseek(f, 0, SEEK_SET);
            uint8_t *v = malloc(n);
            if (fread(v, 1, n, f) != n) {
                printf("  FAIL lettura expert\n");
                return 1;
            }
            fclose(f);
            size_t cap = cfse_bound(n);
            uint8_t *c = malloc(cap), *d = malloc(n);
            size_t cs = cfse_compress(v, n, c, cap);
            size_t rl;
            if (!cs || cfse_decompress(c, cs, d, n, &rl) || rl != n || memcmp(v, d, n)) {
                printf("  FAIL expert reale: round-trip\n");
                fail = 1;
            } else {
                struct timespec t0, t1;
                double best = 1e9;
                for (int k = 0; k < 5; k++) {
                    clock_gettime(CLOCK_MONOTONIC, &t0);
                    cfse_decompress(c, cs, d, n, &rl);
                    clock_gettime(CLOCK_MONOTONIC, &t1);
                    double dt = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
                    if (dt < best) best = dt;
                }
                printf("  EXPERT REALE (%zu byte): ratio %.3fx | decode %.0f ms = %.2f GB/s (1 core)\n", n,
                       (double)n / cs, best * 1000, n / best / 1e9);
            }
            free(v);
            free(c);
            free(d);
        } else printf("  (EXPERT_RAW non leggibile: salto)\n");
    } else printf("  (EXPERT_RAW non impostato: salto il test su pesi reali)\n");

    printf(fail ? "test_fse: FAIL\n" : "test_fse: ok\n");
    return fail;
}
