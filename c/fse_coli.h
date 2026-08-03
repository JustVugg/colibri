/* fse_coli.h — entropy coder di casa per il container colibrì (LOCALE, non pushato).
 *
 * COSA: rANS statico ordine-0 sui NIBBLE (16 simboli), 2 stati interleaved,
 * frequenze normalizzate a 4096 (12 bit), rinormalizzazione a byte.
 * PERCHE' proprio questo: i pesi int4 sono statisticamente BIANCHI (misurato
 * 2026-07-17: condizionali +0.000, MI tra tensori 0.0009-0.0018 bit), quindi
 * l'ordine-0 e' GIA' ottimo — H=2.924 bit/peso, nessun modello di contesto puo'
 * fare meglio, e' un teorema, non una scelta. Ratio atteso ~1.37x, verificato
 * con zstd (stadio FSE) a 1.371x medio su expert reali.
 *
 * SICUREZZA (questo codice tocca i PESI: un bug qui corrompe l'output del
 * modello in silenzio):
 *  - il decoder NON legge mai oltre il buffer: ogni fetch e' bounds-checked;
 *  - lo stato finale dei due rANS DEVE tornare a RANS_L: sigillo d'integrita'
 *    che intercetta troncamenti e corruzioni (non e' un CRC, ma un flip nel
 *    payload disallinea lo stato con probabilita' ~1-2^-46);
 *  - la tabella delle frequenze deve sommare ESATTAMENTE a 4096 o si rifiuta;
 *  - input incomprimibile -> modalita' raw (mai espandere oltre header);
 *  - niente malloc: lavora nei buffer del chiamante.
 *
 * Formato: "CFS1" | mode u8 (0=raw 1=rans) | rawlen u32LE |
 *          mode1: freq[16] u16LE | xA u32LE | xB u32LE | payload
 *          (l'encoder scrive il payload ALL'INDIETRO: il decoder legge in avanti)
 */
#ifndef FSE_COLI_H
#define FSE_COLI_H
#include <stdint.h>
#include <stddef.h>
#include <string.h>

#define CFSE_PROB_BITS 12
#define CFSE_PROB_SCALE (1u << CFSE_PROB_BITS)
#define CFSE_RANS_L (1u << 23)
#define CFSE_HDR 9 /* magic4 + mode1 + rawlen4 */

static inline size_t cfse_bound(size_t n) { return n + CFSE_HDR + 64; }

/* conteggi -> frequenze che sommano ESATTAMENTE a 4096; ogni simbolo presente
 * riceve almeno 1 (un simbolo con freq 0 presente nei dati = encode impossibile). */
static int cfse_normalize(const uint64_t cnt[16], uint16_t freq[16]) {
    uint64_t tot = 0;
    int used = 0;
    for (int i = 0; i < 16; i++) {
        tot += cnt[i];
        if (cnt[i]) used++;
    }
    if (!tot) return -1;
    uint32_t sum = 0;
    int last = -1;
    for (int i = 0; i < 16; i++) {
        if (!cnt[i]) {
            freq[i] = 0;
            continue;
        }
        uint64_t f = (cnt[i] * CFSE_PROB_SCALE) / tot;
        if (!f) f = 1;
        if (f > CFSE_PROB_SCALE - (unsigned)(used - 1)) f = CFSE_PROB_SCALE - (used - 1);
        freq[i] = (uint16_t)f;
        sum += (uint32_t)f;
        last = i;
    }
    /* aggiusta sul simbolo piu' frequente (mai sotto 1) */
    int big = last;
    for (int i = 0; i < 16; i++)
        if (freq[i] > freq[big]) big = i;
    int32_t diff = (int32_t)CFSE_PROB_SCALE - (int32_t)sum;
    if ((int32_t)freq[big] + diff < 1) return -1;
    freq[big] = (uint16_t)((int32_t)freq[big] + diff);
    return 0;
}

/* comprime n byte (2n nibble). Ritorna i byte scritti, 0 = errore/cap corto. */
static size_t cfse_compress(const uint8_t *in, size_t n, uint8_t *out, size_t cap) {
    if (cap < cfse_bound(n) || n > 0xFFFFFFFFu) return 0;
    memcpy(out, "CFS1", 4);
    out[5] = (uint8_t)(n);
    out[6] = (uint8_t)(n >> 8);
    out[7] = (uint8_t)(n >> 16);
    out[8] = (uint8_t)(n >> 24);
    if (n == 0) {
        out[4] = 0;
        return CFSE_HDR;
    }

    uint64_t cnt[16] = {0};
    for (size_t i = 0; i < n; i++) {
        cnt[in[i] & 0xF]++;
        cnt[in[i] >> 4]++;
    }
    uint16_t freq[16];
    uint32_t cum[17] = {0};
    if (cfse_normalize(cnt, freq)) goto raw;
    for (int i = 0; i < 16; i++) cum[i + 1] = cum[i] + freq[i];

    {
        /* encode all'indietro nel fondo di out; poi compattiamo dietro l'header */
        uint8_t *base = out + CFSE_HDR + 32; /* header + tabella freq */
        uint8_t *end = out + cap, *p = end;
        uint32_t xA = CFSE_RANS_L, xB = CFSE_RANS_L;
        size_t N = 2 * n;              /* nibble totali */
        for (size_t i = N; i-- > 0;) { /* i = N-1 .. 0 */
            unsigned s = (i & 1) ? (in[i >> 1] >> 4) : (in[i >> 1] & 0xF);
            uint32_t *x = (i & 1) ? &xB : &xA; /* pari->A, dispari->B (specchio del decode) */
            uint32_t f = freq[s];
            uint32_t xmax = ((CFSE_RANS_L >> CFSE_PROB_BITS) << 8) * f;
            while (*x >= xmax) {
                if (p <= base) goto raw;
                *--p = (uint8_t)(*x & 0xFF);
                *x >>= 8;
            }
            *x = ((*x / f) << CFSE_PROB_BITS) + (*x % f) + cum[s];
        }
        /* flush: B poi A scrivendo all'indietro -> nel flusso in avanti arriva A poi B */
        if (p - base < 8) goto raw;
        for (int k = 3; k >= 0; k--) *--p = (uint8_t)(xB >> (8 * k));
        for (int k = 3; k >= 0; k--) *--p = (uint8_t)(xA >> (8 * k));
        size_t payload = (size_t)(end - p);
        if (CFSE_HDR + 32 + payload >= n + CFSE_HDR) goto raw; /* non conviene */
        out[4] = 1;
        for (int i = 0; i < 16; i++) {
            out[CFSE_HDR + 2 * i] = (uint8_t)freq[i];
            out[CFSE_HDR + 2 * i + 1] = (uint8_t)(freq[i] >> 8);
        }
        memmove(out + CFSE_HDR + 32, p, payload);
        return CFSE_HDR + 32 + payload;
    }
raw:
    out[4] = 0;
    memcpy(out + CFSE_HDR, in, n);
    return CFSE_HDR + n;
}

/* 0 = ok (rawlen scritto), -1 = input invalido/corrotto/troncato. MAI legge
 * oltre in+nin ne' scrive oltre out+cap. */
static int cfse_decompress(const uint8_t *in, size_t nin, uint8_t *out, size_t cap, size_t *rawlen) {
    if (nin < CFSE_HDR || memcmp(in, "CFS1", 4)) return -1;
    size_t n = (size_t)in[5] | ((size_t)in[6] << 8) | ((size_t)in[7] << 16) | ((size_t)in[8] << 24);
    if (n > cap) return -1;
    *rawlen = n;
    if (in[4] == 0) {
        if (nin != CFSE_HDR + n) return -1;
        memcpy(out, in + CFSE_HDR, n);
        return 0;
    }
    if (in[4] != 1 || n == 0 || nin < CFSE_HDR + 32 + 8) return -1;

    uint16_t freq[16];
    uint32_t cum[17] = {0};
    uint32_t sum = 0;
    for (int i = 0; i < 16; i++) {
        freq[i] = (uint16_t)(in[CFSE_HDR + 2 * i] | (in[CFSE_HDR + 2 * i + 1] << 8));
        sum += freq[i];
    }
    if (sum != CFSE_PROB_SCALE) return -1;
    for (int i = 0; i < 16; i++) cum[i + 1] = cum[i] + freq[i];
    /* tabella unica per slot: bias(12b)<<20 | freq(13b)<<7 | sym(4b) = 29 bit.
     * ATTENZIONE al layout: freq puo' valere 4096 (caso un-solo-simbolo) e vuole
     * 13 bit — messa a <<20 overflowava il uint32 e la batteria l'ha beccato.
     * bias = slot - cum[sym]: il passo di decode diventa UNA lookup L1 (16 KB)
     * invece di tre dipendenti (slot2sym -> freq[s] -> cum[s]). */
    uint32_t tab[CFSE_PROB_SCALE];
    for (int s = 0; s < 16; s++)
        for (uint32_t slot = cum[s]; slot < cum[s + 1]; slot++)
            tab[slot] = ((slot - cum[s]) << 20) | ((uint32_t)freq[s] << 7) | (uint32_t)s;

    const uint8_t *p = in + CFSE_HDR + 32, *end = in + nin;
    /* gli stati sono nel flusso in LITTLE-endian (l'encoder li scrive
     * all'indietro byte alto per primo -> in avanti escono LE) */
    uint32_t xA = (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
    uint32_t xB = (uint32_t)p[4] | ((uint32_t)p[5] << 8) | ((uint32_t)p[6] << 16) | ((uint32_t)p[7] << 24);
    p += 8;
    if (xA < CFSE_RANS_L || xB < CFSE_RANS_L) return -1;

    /* Il ciclo caldo lavora a COPPIE (nibble pari->xA, dispari->xB) in registri.
     * Ogni coppia consuma AL MASSIMO 4 byte di payload (2 per stato: da x>=2^11
     * dopo il passo, due rinormalizzazioni a byte riportano sopra 2^23).
     * Quindi: finche' restano >=4*coppie byte, si corre SENZA bounds-check per
     * byte (la matematica lo garantisce, non la fortuna); la coda torna al
     * percorso controllato byte-per-byte. Sicurezza invariata: stesso formato,
     * stessi sigilli (p==end, stati==RANS_L), la batteria + ASAN lo certificano. */
    uint32_t xa = xA, xb = xB;
    size_t pairs = n, j = 0;
    while (j < pairs) {
        size_t safe = (size_t)(end - p) / 4; /* coppie garantite senza check */
        size_t stop = j + (pairs - j < safe ? pairs - j : safe);
        for (; j < stop; j++) {
            uint32_t ea = tab[xa & (CFSE_PROB_SCALE - 1)];
            xa = ((ea >> 7) & 0x1FFF) * (xa >> CFSE_PROB_BITS) + (ea >> 20);
            if (xa < CFSE_RANS_L) {
                xa = (xa << 8) | *p++;
                if (xa < CFSE_RANS_L) xa = (xa << 8) | *p++;
            }
            uint32_t eb = tab[xb & (CFSE_PROB_SCALE - 1)];
            xb = ((eb >> 7) & 0x1FFF) * (xb >> CFSE_PROB_BITS) + (eb >> 20);
            if (xb < CFSE_RANS_L) {
                xb = (xb << 8) | *p++;
                if (xb < CFSE_RANS_L) xb = (xb << 8) | *p++;
            }
            out[j] = (uint8_t)((ea & 0xF) | ((eb & 0xF) << 4));
        }
        if (j < pairs) { /* coda: percorso CONTROLLATO */
            uint32_t ea = tab[xa & (CFSE_PROB_SCALE - 1)];
            xa = ((ea >> 7) & 0x1FFF) * (xa >> CFSE_PROB_BITS) + (ea >> 20);
            while (xa < CFSE_RANS_L) {
                if (p >= end) return -1;
                xa = (xa << 8) | *p++;
            }
            uint32_t eb = tab[xb & (CFSE_PROB_SCALE - 1)];
            xb = ((eb >> 7) & 0x1FFF) * (xb >> CFSE_PROB_BITS) + (eb >> 20);
            while (xb < CFSE_RANS_L) {
                if (p >= end) return -1;
                xb = (xb << 8) | *p++;
            }
            out[j] = (uint8_t)((ea & 0xF) | ((eb & 0xF) << 4));
            j++;
        }
    }
    xA = xa;
    xB = xb;                                               /* per i sigilli finali */
    if (p != end) return -1;                               /* byte avanzati = flusso non nostro */
    if (xA != CFSE_RANS_L || xB != CFSE_RANS_L) return -1; /* sigillo d'integrita' */
    return 0;
}
#endif
