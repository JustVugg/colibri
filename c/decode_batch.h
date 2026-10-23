#ifndef COLIBRI_DECODE_BATCH_H
#define COLIBRI_DECODE_BATCH_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

/* `base` belongs to one sequence's KV state.  Keeping this arithmetic in a
 * model-independent seam makes ragged decode row ownership directly testable. */
static inline float *coli_kv_row(float *base, int position, int width)
{
    return base + (size_t)position * (size_t)width;
}

/* Per-token top-k emission cap: run_ablate_score's existing top-32 read-out
 * ceiling, adopted as the wire cap too (no named consumer asks for more). */
#define COLI_SUBMIT_TOPK_MAX 32

/* KV8 twin: same row arithmetic on the fp8 (e4m3) byte cache. */
static inline uint8_t *coli_kv_row8(uint8_t *base, int position, int width)
{
    return base + (size_t)position * (size_t)width;
}

typedef struct {
    unsigned long long id, bytes, gbytes;
    int slot, max_tokens;
    int pin;        /* SUBMIT pin=1: fotografa lo stato dopo il prefill di questa
                     * richiesta, cosi le successive che cominciano con lo stesso
                     * prompt possono riavvolgersi invece di rifarlo. Serve a chi
                     * punteggia un menu chiuso: lo stato condiviso si calcola una
                     * volta e ogni opzione paga solo i propri token. */
    float temperature, top_p;
    int logprobs;   /* requested per-token top-k count; 0 = channel off (opt-in) */
    int tok_ids;    /* 1 = payload is ASCII token ids, not raw prompt text */
} ColiSubmit;

/* Extension fields, 8th whitespace field onward: "key=value" tokens after the
 * full 7-field header, e.g. "SUBMIT 1 0 12 16 0 1 0 logprobs=5 ids=1".
 * A namespace instead of more positional fields: the next per-request knob
 * (a future seed, say) claims a key here rather than forcing another
 * field-count migration of this scarce wire budget. Unknown keys, duplicate
 * keys, glued tokens and out-of-range values all reject the whole frame --
 * framing stays as unambiguous as the legacy arms' trailing-field guard.
 * An OLD engine rejects ANY extended header: both legacy sscanf arms fail on
 * the extra field (their trailing %c matches the separating space) and the
 * engine answers ERROR 0 BAD_REQUEST. The rejection itself is safe, but the
 * reject path does not drain the rejected request's payload bytes, so an old
 * engine re-reads them as protocol lines -- the pre-existing behavior of
 * every malformed-SUBMIT reject, not exposure added by this namespace.
 *
 * The value is accumulated by hand into a bounded int (digit loop stops once
 * the running value exceeds the largest legal value of any key, then a
 * pending-digit check rejects): sscanf %llu on an overflowing field is
 * undefined behavior (C11 7.21.6.2), so no sscanf ever touches the value. */
static inline int coli_submit_ext(const char *p, ColiSubmit *s)
{
    int seen = 0, seen_logprobs = 0, seen_ids = 0, seen_pin = 0;
    while (*p == ' ' || *p == '\t') p++;
    while (*p) {
        char key[16];
        int kn = 0, val = 0;
        const char *d;
        while (kn < 15 && ((*p >= 'a' && *p <= 'z') || *p == '_'))
            key[kn++] = *p++;
        key[kn] = 0;
        if (kn == 0 || *p != '=') return 0;
        p++;
        d = p;
        while (*p >= '0' && *p <= '9' && val <= COLI_SUBMIT_TOPK_MAX)
            val = val * 10 + (*p++ - '0');
        if (p == d) return 0;                        /* no digits after '=' */
        if (*p >= '0' && *p <= '9') return 0;        /* out of range: digits left */
        if (*p && *p != ' ' && *p != '\t') return 0; /* glued garbage after value */
        if (!strcmp(key, "logprobs")) {          /* per-token top-k emission */
            if (seen_logprobs || val > COLI_SUBMIT_TOPK_MAX) return 0;
            seen_logprobs = 1;
            s->logprobs = val;
        } else if (!strcmp(key, "ids")) {        /* pre-tokenized prompt intake */
            if (seen_ids || val > 1) return 0;
            seen_ids = 1;
            s->tok_ids = val;
        } else if (!strcmp(key, "pin")) {        /* fotografa lo stato dopo il prefill */
            if (seen_pin || val > 1) return 0;
            seen_pin = 1;
            s->pin = val;
        } else {
            return 0;                            /* unknown key */
        }
        seen = 1;
        while (*p == ' ' || *p == '\t') p++;
    }
    return seen;
}

/* Parse the textual header. The payload is read separately using `bytes`, so
 * it may contain newlines. Reject trailing fields to keep framing unambiguous.
 * Optional 7th field `gbytes`: length of a per-request grammar (raw GBNF, or a
 * JSON-Schema compiled engine-side) appended to the payload AFTER the prompt
 * bytes. 6-field headers remain valid (gbytes = 0). Fields past the 7th are
 * key=value extension tokens (coli_submit_ext above), only ever sent by a
 * server that knows this engine understands them. */
static inline int coli_submit_parse(const char *line, ColiSubmit *s)
{
    char tail;
    int ok = 0, base = 0;
    if (!line || !s) return 0;
    s->gbytes = 0;
    s->logprobs = 0;
    s->tok_ids = 0;
    s->pin = 0;
    if (sscanf(line, "SUBMIT %llu %d %llu %d %f %f %llu %c", &s->id, &s->slot,
               &s->bytes, &s->max_tokens, &s->temperature, &s->top_p,
               &s->gbytes, &tail) == 7) ok = 1;
    if (!ok) {
        s->gbytes = 0;
        if (sscanf(line, "SUBMIT %llu %d %llu %d %f %f %c", &s->id, &s->slot,
                   &s->bytes, &s->max_tokens, &s->temperature, &s->top_p,
                   &tail) == 6) ok = 1;
    }
    if (!ok) {
        s->gbytes = 0;
        /* Extended arm: the numeric gbytes field and the first key=value token
         * MUST be whitespace-separated ("512logprobs=5" is one malformed field,
         * not a 512 followed by an opt-in -- %llu would otherwise stop at the
         * first letter and silently split it). */
        if (sscanf(line, "SUBMIT %llu %d %llu %d %f %f %llu%n", &s->id,
                   &s->slot, &s->bytes, &s->max_tokens, &s->temperature,
                   &s->top_p, &s->gbytes, &base) == 7 && base > 0 &&
            (line[base] == ' ' || line[base] == '\t') &&
            coli_submit_ext(line + base, s)) ok = 1;
    }
    if (!ok) return 0;
    return s->id > 0 && s->bytes <= (16u << 20) && s->gbytes <= (1u << 20) &&
           s->slot >= 0 &&
           /* max_tokens=0: legittimo solo con logprobs>0 (vedi serve_codec.h) */
           (s->max_tokens >= 1 || (s->max_tokens == 0 && s->logprobs > 0)) &&
           isfinite(s->temperature) && isfinite(s->top_p) &&
           s->temperature >= 0 && s->temperature <= 2 &&
           s->top_p > 0 && s->top_p <= 1;
}

/* SUBMIT ids=1 payload: ASCII decimal token ids separated by whitespace,
 * parsed straight into the prompt buffer -- the pre-tokenized intake path
 * that bypasses tok_encode entirely (no detokenize/re-encode round trip, so
 * the ids the caller sent are exactly the ids the engine scores). Returns
 * the id count, or -1 on any malformed or out-of-range id. More than `cap`
 * ids reports `cap`: the caller parses with one spare slot and treats a full
 * buffer as overflow, mirroring tok_encode's stop-at-cap contract (#401). */
static inline int coli_ids_parse(const char *buf, size_t len, int *out,
                                 int cap, int vocab)
{
    size_t i = 0;
    int n = 0;
    if (!buf || !out || cap < 1 || vocab < 1) return -1;
    while (i < len) {
        while (i < len && (unsigned char)buf[i] <= ' ') i++;
        if (i >= len) break;
        if (n >= cap) return cap;            /* overflow: caller refuses loudly */
        {
            long v = 0;
            size_t d = i;
            while (i < len && buf[i] >= '0' && buf[i] <= '9' && v < vocab)
                v = v * 10 + (buf[i++] - '0');
            if (i == d || v >= vocab || (i < len && (unsigned char)buf[i] > ' '))
                return -1;
            out[n++] = (int)v;
        }
    }
    return n;
}

/* Coda numerica di un token: " <lp> <k> [tid tlp]*k", in log-softmax sul
 * vocabolario intero. Vive qui, accanto a COLI_SUBMIT_TOPK_MAX e al parser
 * delle chiavi, perche e l'altra meta dello stesso canale: chi lo legge e chi
 * lo scrive devono concordare sul formato, e un formato concordato in due
 * copie non e concordato. colibri.c ha la propria logprob_tail dal giorno in
 * cui il canale e nato; questa ne e la versione condivisa per gli altri motori
 * e produce gli stessi byte.
 *
 * lo==NULL, oppure un token fuori dal vocabolario, danno " nan 0": e il caso
 * della posizione 0 di un echo, dove non c'e nulla su cui condizionare.
 * Costo: O(V) per la normalizzazione e O(V*k) per la selezione, pagato solo
 * dalle richieste che hanno chiesto il canale. */
static inline int coli_logprob_tail(char *dst, size_t cap, const float *lo,
                                    int V, int token, int topk)
{
    int tk_id[COLI_SUBMIT_TOPK_MAX];
    float tk_val[COLI_SUBMIT_TOPK_MAX];
    double se = 0, logZ;
    float mx;
    int w, i, k;
    if (!dst || cap == 0) return 0;
    if (!lo || token < 0 || token >= V || topk <= 0)
        return snprintf(dst, cap, " nan 0");
    if (topk > COLI_SUBMIT_TOPK_MAX) topk = COLI_SUBMIT_TOPK_MAX;
    if (topk > V) topk = V;
    mx = lo[0];
    for (i = 1; i < V; i++) if (lo[i] > mx) mx = lo[i];
    for (i = 0; i < V; i++) se += exp((double)lo[i] - mx);
    logZ = (double)mx + log(se);
    for (k = 0; k < topk; k++) { tk_id[k] = -1; tk_val[k] = -1e30f; }
    for (i = 0; i < V; i++) {
        float v = lo[i];
        int mn = 0;
        for (k = 1; k < topk; k++) if (tk_val[k] < tk_val[mn]) mn = k;
        if (v > tk_val[mn]) { tk_val[mn] = v; tk_id[mn] = i; }
    }
    w = snprintf(dst, cap, " %.6f %d", (double)lo[token] - logZ, topk);
    for (k = 0; k < topk && w > 0 && (size_t)w < cap; k++)
        w += snprintf(dst + w, cap - (size_t)w, " %d %.6f",
                      tk_id[k], (double)tk_val[k] - logZ);
    return w;
}

#endif
