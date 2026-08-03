/* cfse_pack — converte uno shard safetensors nel container entropy-coded CFSE
 * (LOCALE, non pushato). E lo VERIFICA: --verify rilegge il convertito, lo
 * decomprime e lo confronta byte-per-byte con l'originale.
 *
 * Formato d'uscita: safetensors normale, ma:
 *   - __metadata__ contiene "cfse":"1"
 *   - il payload di OGNI tensore e' un flusso CFS1 (fse_coli.h), col fallback
 *     raw interno per i tensori incomprimibili
 *   - data_offsets copre l'estensione COMPRESSA; la taglia raw resta
 *     ricostruibile da shape x dtype (invariante safetensors) e dal campo
 *     rawlen dentro CFS1 (doppio controllo in lettura)
 *
 * Uso:  cfse_pack in.safetensors out.safetensors        converte
 *       cfse_pack --verify in.safetensors out.safetensors   certifica il convertito
 *       cfse_pack --cert in.safetensors                     certificazione IN MEMORIA:
 *         round-trip compress+decompress+memcmp di OGNI tensore, per-tensore via
 *         fseek (RAM ~2x il tensore piu' grande, mai il file intero), ZERO scritture.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "json.h"
#include "fse_coli.h"

static void *xmalloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) {
        fprintf(stderr, "OOM %zu\n", n);
        exit(1);
    }
    return p;
}

typedef struct {
    const char *name;
    const char *dtype;
    jval *shape;
    int64_t a, b;
} TEnt;

static int cmp_off(const void *x, const void *y) {
    const TEnt *A = x, *B = y;
    return (A->a > B->a) - (A->a < B->a);
}

static char *read_file(const char *path, size_t *n) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        perror(path);
        exit(1);
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *b = xmalloc((size_t)sz);
    if (fread(b, 1, (size_t)sz, f) != (size_t)sz) {
        perror("fread");
        exit(1);
    }
    fclose(f);
    *n = (size_t)sz;
    return b;
}

static int parse_shard(char *buf, size_t n, jval **root_out, char **arena_out, TEnt **ents_out, int *nents_out,
                       size_t *data_start_out) {
    if (n < 8) return -1;
    uint64_t hlen;
    memcpy(&hlen, buf, 8);
    if (hlen > n - 8) return -1;
    char *hdr = xmalloc(hlen + 1);
    memcpy(hdr, buf + 8, hlen);
    hdr[hlen] = 0;
    char *arena = NULL;
    jval *root = json_parse(hdr, &arena);
    if (!root || root->t != J_OBJ) return -1;
    TEnt *ents = xmalloc(sizeof(TEnt) * (size_t)root->len);
    int ne = 0;
    for (int i = 0; i < root->len; i++) {
        if (!strcmp(root->keys[i], "__metadata__")) continue;
        jval *m = root->kids[i];
        jval *dt = json_get(m, "dtype"), *off = json_get(m, "data_offsets"), *shp = json_get(m, "shape");
        if (!dt || dt->t != J_STR || !off || off->t != J_ARR || off->len < 2 || !shp || shp->t != J_ARR) return -1;
        ents[ne].name = root->keys[i];
        ents[ne].dtype = dt->str;
        ents[ne].shape = shp;
        ents[ne].a = (int64_t)off->kids[0]->num;
        ents[ne].b = (int64_t)off->kids[1]->num;
        ne++;
    }
    qsort(ents, (size_t)ne, sizeof(TEnt), cmp_off);
    *root_out = root;
    *arena_out = arena;
    *ents_out = ents;
    *nents_out = ne;
    *data_start_out = 8 + hlen;
    (void)hdr; /* le stringhe json puntano dentro hdr: resta vivo */
    return 0;
}

static void emit_json_shape(FILE *o, jval *shp) {
    fputc('[', o);
    for (int k = 0; k < shp->len; k++) fprintf(o, "%s%lld", k ? "," : "", (long long)shp->kids[k]->num);
    fputc(']', o);
}

static int do_cert(const char *inp) {
    FILE *f = fopen(inp, "rb");
    if (!f) {
        perror(inp);
        return 1;
    }
    uint64_t hlen;
    if (fread(&hlen, 8, 1, f) != 1) {
        fprintf(stderr, "%s: header\n", inp);
        return 1;
    }
    char *hdr = xmalloc(hlen + 1);
    if (fread(hdr, 1, hlen, f) != hlen) {
        fprintf(stderr, "%s: header troncato\n", inp);
        return 1;
    }
    hdr[hlen] = 0;
    char *arena = NULL;
    jval *root = json_parse(hdr, &arena);
    if (!root || root->t != J_OBJ) {
        fprintf(stderr, "%s: json\n", inp);
        return 1;
    }
    size_t ds = 8 + hlen;
    int nt = 0;
    int64_t rawtot = 0, comptot = 0;
    for (int i = 0; i < root->len; i++) {
        if (!strcmp(root->keys[i], "__metadata__")) continue;
        jval *m = root->kids[i];
        jval *off = json_get(m, "data_offsets");
        if (!off || off->t != J_ARR || off->len < 2) {
            fprintf(stderr, "%s: offsets %s\n", inp, root->keys[i]);
            return 1;
        }
        size_t a = (size_t)off->kids[0]->num, b = (size_t)off->kids[1]->num, rn = b - a;
        uint8_t *raw = xmalloc(rn ? rn : 1), *c = xmalloc(cfse_bound(rn)), *d = xmalloc(rn ? rn : 1);
        if (fseek(f, (long)(ds + a), SEEK_SET) || fread(raw, 1, rn, f) != rn) {
            fprintf(stderr, "%s: read %s\n", inp, root->keys[i]);
            return 1;
        }
        size_t cs = cfse_compress(raw, rn, c, cfse_bound(rn));
        size_t rl = 0;
        if (!cs || cfse_decompress(c, cs, d, rn, &rl) || rl != rn || (rn && memcmp(raw, d, rn))) {
            fprintf(stderr, "CERT FAIL: %s / %s\n", inp, root->keys[i]);
            return 1;
        }
        free(raw);
        free(c);
        free(d);
        nt++;
        rawtot += (int64_t)rn;
        comptot += (int64_t)cs;
    }
    fclose(f);
    printf("CERT OK %s: %d tensori | %.3fx\n", inp, nt, comptot ? (double)rawtot / (double)comptot : 1.0);
    return 0;
}

int main(int argc, char **argv) {
    if (argc == 3 && !strcmp(argv[1], "--cert")) return do_cert(argv[2]);
    int verify = argc > 1 && !strcmp(argv[1], "--verify");
    if ((verify && argc != 4) || (!verify && argc != 3)) {
        fprintf(stderr, "uso: %s in.st out.st | %s --verify in.st out.st\n", argv[0], argv[0]);
        return 2;
    }
    const char *inp = argv[verify ? 2 : 1], *outp = argv[verify ? 3 : 2];

    size_t inN;
    char *in = read_file(inp, &inN);
    jval *root;
    char *arena;
    TEnt *E;
    int NE;
    size_t ds;
    if (parse_shard(in, inN, &root, &arena, &E, &NE, &ds)) {
        fprintf(stderr, "%s: header non valido\n", inp);
        return 1;
    }

    if (verify) {
        size_t oN;
        char *ob = read_file(outp, &oN);
        jval *oroot;
        char *oarena;
        TEnt *OE;
        int ONE;
        size_t ods;
        if (parse_shard(ob, oN, &oroot, &oarena, &OE, &ONE, &ods)) {
            fprintf(stderr, "%s: header non valido\n", outp);
            return 1;
        }
        if (ONE != NE) {
            fprintf(stderr, "VERIFY FAIL: %d vs %d tensori\n", NE, ONE);
            return 1;
        }
        int64_t rawtot = 0, comptot = 0;
        for (int i = 0; i < NE; i++) {
            /* trova il gemello per nome (l'ordine per offset puo' differire) */
            TEnt *o = NULL;
            for (int j = 0; j < ONE; j++)
                if (!strcmp(OE[j].name, E[i].name)) {
                    o = &OE[j];
                    break;
                }
            if (!o) {
                fprintf(stderr, "VERIFY FAIL: manca %s\n", E[i].name);
                return 1;
            }
            size_t rn = (size_t)(E[i].b - E[i].a), cn = (size_t)(o->b - o->a);
            uint8_t *dec = xmalloc(rn);
            size_t rl = 0;
            if (cfse_decompress((uint8_t *)ob + ods + o->a, cn, dec, rn, &rl) || rl != rn ||
                memcmp(dec, in + ds + E[i].a, rn)) {
                fprintf(stderr, "VERIFY FAIL: %s non identico dopo il round-trip\n", E[i].name);
                return 1;
            }
            free(dec);
            rawtot += (int64_t)rn;
            comptot += (int64_t)cn;
        }
        printf("VERIFY OK: %d tensori BIT-IDENTICI | %lld -> %lld byte (%.3fx)\n", NE, (long long)rawtot,
               (long long)comptot, (double)rawtot / (double)comptot);
        return 0;
    }

    /* --- conversione: comprimi ogni tensore, ricostruisci header con nuovi offset --- */
    uint8_t **blob = xmalloc(sizeof(void *) * (size_t)NE);
    size_t *bn = xmalloc(sizeof(size_t) * (size_t)NE);
    int64_t rawtot = 0, comptot = 0;
    for (int i = 0; i < NE; i++) {
        size_t rn = (size_t)(E[i].b - E[i].a);
        blob[i] = xmalloc(cfse_bound(rn));
        bn[i] = cfse_compress((uint8_t *)in + ds + E[i].a, rn, blob[i], cfse_bound(rn));
        if (!bn[i]) {
            fprintf(stderr, "%s: compress fallita su %s\n", inp, E[i].name);
            return 1;
        }
        /* paranoia in linea: round-trip IMMEDIATO prima di scrivere qualsiasi cosa */
        uint8_t *chk = xmalloc(rn);
        size_t rl = 0;
        if (cfse_decompress(blob[i], bn[i], chk, rn, &rl) || rl != rn || memcmp(chk, in + ds + E[i].a, rn)) {
            fprintf(stderr, "ABORT: self-check fallito su %s — nessun file scritto\n", E[i].name);
            return 1;
        }
        free(chk);
        rawtot += (int64_t)rn;
        comptot += (int64_t)bn[i];
    }

    /* header JSON: __metadata__.cfse=1 + tensori con offset compressi (ordine originale) */
    FILE *o = fopen(outp, "wb");
    if (!o) {
        perror(outp);
        return 1;
    }
    fseek(o, 8, SEEK_SET); /* hlen scritto alla fine */
    long h0 = ftell(o);
    fprintf(o, "{\"__metadata__\":{\"cfse\":\"1\"}");
    int64_t cur = 0;
    for (int i = 0; i < NE; i++) {
        fprintf(o, ",\"%s\":{\"dtype\":\"%s\",\"shape\":", E[i].name, E[i].dtype);
        emit_json_shape(o, E[i].shape);
        fprintf(o, ",\"data_offsets\":[%lld,%lld]}", (long long)cur, (long long)(cur + (int64_t)bn[i]));
        cur += (int64_t)bn[i];
    }
    fputc('}', o);
    long h1 = ftell(o);
    while ((h1 - h0) % 8) {
        fputc(' ', o);
        h1++;
    } /* padding a 8 per buona educazione */
    uint64_t hlen = (uint64_t)(h1 - h0);
    for (int i = 0; i < NE; i++)
        if (fwrite(blob[i], 1, bn[i], o) != bn[i]) {
            perror("fwrite");
            return 1;
        }
    fseek(o, 0, SEEK_SET);
    fwrite(&hlen, 8, 1, o);
    fclose(o);
    printf("%s: %d tensori | %lld -> %lld byte (%.3fx)\n", outp, NE, (long long)rawtot, (long long)comptot,
           (double)rawtot / (double)comptot);
    return 0;
}
