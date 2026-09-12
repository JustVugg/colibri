/* Parser JSON minimale, header-only. Serve per:
 *  - l'header dei file safetensors (un grande oggetto nome->{dtype,shape,data_offsets})
 *  - ref.json (per leggere prompt_ids / full_ids)
 * Non e' completo (niente unicode \uXXXX, niente notazione esotica) ma copre cio' che serve. */
#ifndef JSON_H
#define JSON_H
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <ctype.h>
#include <math.h>

typedef enum { J_NULL, J_BOOL, J_NUM, J_STR, J_ARR, J_OBJ } jtype;

typedef struct jval {
    jtype t;
    double num;            /* J_NUM */
    int    boolean;        /* J_BOOL */
    char  *str;            /* J_STR (NUL-terminata, dentro l'arena) */
    /* array: figli in [0..len); oggetto: chiavi[] e figli[] in parallelo */
    struct jval **kids;
    char        **keys;    /* solo per J_OBJ */
    int           len;
} jval;

typedef struct {
    const char *s;
    char       *arena;     /* buffer per le stringhe smontate */
    size_t      acap, aoff;
    int         depth;     /* annidamento corrente: bound contro lo stack-overflow
                            * da JSON malevolo tipo [[[[...]]]] (discesa ricorsiva) */
    int         strict;    /* grammatica esatta; rifiuta NUL per stringhe C sicure */
    int         error;
} jparser;

/* tetto di annidamento: gli header safetensors / config sono piatti (profondita'
 * ~3). 1024 e' larghissimo per input legittimi e ben sotto il limite di stack. */
#define J_MAX_DEPTH 1024

static char *j_dup(jparser *p, const char *b, int n) {
    /* ogni stringa ha la sua allocazione: un'arena con realloc sposterebbe il
     * buffer invalidando i puntatori gia' emessi (use-after-free). */
    (void)p;
    char *d = (char *)malloc(n + 1);
    memcpy(d, b, n); d[n] = 0;
    return d;
}

static void j_ws(jparser *p) {
    if (p->strict) {
        while (*p->s == ' ' || *p->s == '\t' ||
               *p->s == '\n' || *p->s == '\r') p->s++;
    } else {
        while (*p->s && isspace((unsigned char)*p->s)) p->s++;
    }
}

static jval *j_new(jtype t) {
    jval *v = (jval *)calloc(1, sizeof(jval));
    v->t = t; return v;
}

static jval *j_parse_val(jparser *p);

static int j_hex4(const char *s, unsigned *out) {
    unsigned value = 0;
    for (int i = 0; i < 4; i++) {
        unsigned digit;
        unsigned char c = (unsigned char)s[i];
        if (c >= '0' && c <= '9') digit = c - '0';
        else if (c >= 'a' && c <= 'f') digit = c - 'a' + 10;
        else if (c >= 'A' && c <= 'F') digit = c - 'A' + 10;
        else return -1;
        value = (value << 4) | digit;
    }
    *out = value;
    return 0;
}

static char *j_parse_str_raw(jparser *p) {
    /* SEC (GHSA-2qrj): fail closed if not actually at a quote. The old comment
     * "assume *p->s == '\"'" was violated on the object-key path, and the
     * unconditional p->s++ would step past the buffer's NUL terminator and scan
     * adjacent heap (OOB read leaking into tensor names). */
    if (*p->s != '"') {
        if (p->strict) p->error = 1;
        return j_dup(p, "", 0);
    }
    p->s++;
    /* buffer su heap che CRESCE: niente troncamento silenzioso a 64KB (le stringhe
     * lunghe di tokenizer.json/config venivano tagliate) e niente 64KB di stack. */
    size_t cap = 64, n = 0; char *tmp = (char *)malloc(cap);
    if (!tmp) { fprintf(stderr, "OOM parsing JSON string\n"); exit(1); }
    #define J_PUT(ch) do{ if (n + 1 >= cap) { cap *= 2; tmp = (char *)realloc(tmp, cap); \
        if (!tmp) { fprintf(stderr, "OOM parsing JSON string\n"); exit(1); } } tmp[n++] = (char)(ch); }while(0)
    while (*p->s && *p->s != '"') {
        char c = *p->s++;
        if (p->strict && (unsigned char)c < 0x20) p->error = 1;
        if (c == '\\' && !*p->s) {
            if (p->strict) p->error = 1;
        } else if (c == '\\') {
            char e = *p->s++;
            switch (e) {
                case 'n': c = '\n'; break; case 't': c = '\t'; break;
                case 'r': c = '\r'; break; case 'b': c = '\b'; break;
                case 'f': c = '\f'; break; case '/': c = '/'; break;
                case '\\': c = '\\'; break; case '"': c = '"'; break;
                case 'u': {  /* \uXXXX -> codepoint UTF-8 (con coppie surrogate) */
                    if (!p->s[0]||!p->s[1]||!p->s[2]||!p->s[3]) {
                        if (p->strict) p->error = 1;
                        c='?'; break;   /* \u troncato: non leggere oltre il NUL */
                    }
                    unsigned cp = (unsigned)strtoul((char[]){p->s[0],p->s[1],p->s[2],p->s[3],0}, NULL, 16);
                    if (p->strict && j_hex4(p->s, &cp)) p->error = 1;
                    p->s += 4;
                    if (p->strict && cp == 0) p->error = 1;
                    if (cp >= 0xD800 && cp <= 0xDBFF) {
                        unsigned lo = 0;
                        int has_low = p->s[0]=='\\' && p->s[1]=='u' &&
                            p->s[2] && p->s[3] && p->s[4] && p->s[5] &&
                            j_hex4(p->s + 2, &lo) == 0 &&
                            lo >= 0xDC00 && lo <= 0xDFFF;
                        if (has_low) {
                            cp = 0x10000 + ((cp-0xD800)<<10) + (lo-0xDC00);
                            p->s += 6;
                        } else if (p->strict) {
                            p->error = 1;
                        }
                    } else if (p->strict && cp >= 0xDC00 && cp <= 0xDFFF) {
                        p->error = 1;
                    }
                    if (cp < 0x80) { J_PUT(cp); }
                    else if (cp < 0x800) { J_PUT(0xC0|(cp>>6)); J_PUT(0x80|(cp&0x3F)); }
                    else if (cp < 0x10000) { J_PUT(0xE0|(cp>>12)); J_PUT(0x80|((cp>>6)&0x3F)); J_PUT(0x80|(cp&0x3F)); }
                    else { J_PUT(0xF0|(cp>>18)); J_PUT(0x80|((cp>>12)&0x3F)); J_PUT(0x80|((cp>>6)&0x3F)); J_PUT(0x80|(cp&0x3F)); }
                    continue;
                }
                default: if (p->strict) p->error = 1; c = e; break;
            }
        }
        J_PUT(c);
    }
    #undef J_PUT
    if (*p->s == '"') p->s++;
    else if (p->strict) p->error = 1;
    char *out = j_dup(p, tmp, (int)n); free(tmp);
    return out;
}

static jval *j_parse_val(jparser *p) {
    j_ws(p);
    char c = *p->s;
    if (c == '"') { jval *v = j_new(J_STR); v->str = j_parse_str_raw(p); return v; }
    if (c == '{') {
        if (++p->depth > J_MAX_DEPTH) {
            if (p->strict) p->error = 1;
            p->depth--;
            return j_new(J_NULL);
        }
        p->s++; jval *v = j_new(J_OBJ);
        int cap = 8;
        v->keys = malloc(cap * sizeof(char*));
        if (!v->keys) { fprintf(stderr, "OOM parsing JSON object\n"); exit(1); }
        v->kids = malloc(cap * sizeof(jval*));
        if (!v->kids) { fprintf(stderr, "OOM parsing JSON object\n"); exit(1); }
        j_ws(p);
        if (*p->s == '}') { p->s++; p->depth--; return v; }
        int closed = 0;
        for (;;) {
            j_ws(p);
            if (*p->s != '"') {
                if (p->strict) p->error = 1;
                break;   /* SEC (GHSA-2qrj): object key must be a quoted string; stop on malformed input */
            }
            char *key = j_parse_str_raw(p);
            j_ws(p);
            if (*p->s == ':') p->s++;
            else if (p->strict) p->error = 1;
            jval *val = j_parse_val(p);
            if (v->len == cap) { cap *= 2;
                char **nk = (char**)realloc(v->keys, cap*sizeof(char*));
                if (!nk) { fprintf(stderr, "OOM parsing JSON object\n"); exit(1); }
                v->keys = nk;
                jval **nc = (jval**)realloc(v->kids, cap*sizeof(jval*));
                if (!nc) { fprintf(stderr, "OOM parsing JSON object\n"); exit(1); }
                v->kids = nc; }
            v->keys[v->len] = key; v->kids[v->len] = val; v->len++;
            j_ws(p);
            if (*p->s == ',') {
                p->s++;
                if (p->strict) {
                    j_ws(p);
                    if (*p->s == '}') p->error = 1;
                }
                continue;
            }
            if (*p->s == '}') { p->s++; closed = 1; break; }
            if (p->strict) p->error = 1;
            break;
        }
        if (p->strict && !closed) p->error = 1;
        p->depth--;
        return v;
    }
    if (c == '[') {
        if (++p->depth > J_MAX_DEPTH) {
            if (p->strict) p->error = 1;
            p->depth--;
            return j_new(J_NULL);
        }
        p->s++; jval *v = j_new(J_ARR);
        int cap = 8;
        v->kids = malloc(cap * sizeof(jval*));
        if (!v->kids) { fprintf(stderr, "OOM parsing JSON array\n"); exit(1); }
        j_ws(p);
        if (*p->s == ']') { p->s++; p->depth--; return v; }
        int closed = 0;
        for (;;) {
            if (p->strict) {
                j_ws(p);
                if (*p->s == ']') {
                    p->error = 1;
                    p->s++;
                    closed = 1;
                    break;
                }
            }
            jval *val = j_parse_val(p);
            if (v->len == cap) { cap *= 2;
                jval **nc = (jval**)realloc(v->kids, cap*sizeof(jval*));
                if (!nc) { fprintf(stderr, "OOM parsing JSON array\n"); exit(1); }
                v->kids = nc; }
            v->kids[v->len++] = val;
            j_ws(p);
            if (*p->s == ',') { p->s++; continue; }
            if (*p->s == ']') { p->s++; closed = 1; break; }
            if (p->strict) p->error = 1;
            break;
        }
        if (p->strict && !closed) p->error = 1;
        p->depth--;
        return v;
    }
    if (c == 't' && !strncmp(p->s, "true", 4))  { p->s += 4; jval *v = j_new(J_BOOL); v->boolean = 1; return v; }
    if (c == 'f' && !strncmp(p->s, "false", 5)) { p->s += 5; jval *v = j_new(J_BOOL); v->boolean = 0; return v; }
    if (c == 'n' && !strncmp(p->s, "null", 4))  { p->s += 4; return j_new(J_NULL); }
    /* numero */
    if (p->strict) {
        const char *start = p->s;
        const char *scan = start;
        if (*scan == '-') scan++;
        if (*scan == '0') {
            scan++;
            if (isdigit((unsigned char)*scan)) p->error = 1;
        } else if (*scan >= '1' && *scan <= '9') {
            do { scan++; } while (isdigit((unsigned char)*scan));
        } else {
            p->error = 1;
        }
        if (*scan == '.') {
            scan++;
            if (!isdigit((unsigned char)*scan)) p->error = 1;
            while (isdigit((unsigned char)*scan)) scan++;
        }
        if (*scan == 'e' || *scan == 'E') {
            scan++;
            if (*scan == '+' || *scan == '-') scan++;
            if (!isdigit((unsigned char)*scan)) p->error = 1;
            while (isdigit((unsigned char)*scan)) scan++;
        }
        char *end;
        double d = strtod(start, &end);
        if (end != scan || end == start || !isfinite(d)) p->error = 1;
        p->s = scan;
        jval *v = j_new(J_NUM); v->num = d; return v;
    }
    { char *end; double d = strtod(p->s, &end); p->s = end; jval *v = j_new(J_NUM); v->num = d; return v; }
}

/* API */
static inline jval *json_parse(const char *text, char **arena_out) {
    jparser p = { text, NULL, 0, 0, 0, 0, 0 };
    jval *v = j_parse_val(&p);
    if (arena_out) *arena_out = p.arena; else free(p.arena);
    return v;
}

static jval *json_get(jval *o, const char *key) {
    if (!o || o->t != J_OBJ) return NULL;
    for (int i = 0; i < o->len; i++) if (strcmp(o->keys[i], key) == 0) return o->kids[i];
    return NULL;
}

static void json_free(jval *v) {
    if (!v) return;
    if (v->t == J_ARR || v->t == J_OBJ) {
        for (int i = 0; i < v->len; i++) {
            json_free(v->kids[i]);
            if (v->t == J_OBJ) free(v->keys[i]);
        }
        free(v->kids);
        free(v->keys);
    } else if (v->t == J_STR) {
        free(v->str);
    }
    free(v);
}

/* Exact parsing is opt-in so existing safetensors/config callers retain the
 * historical permissive behavior. It rejects malformed syntax, trailing
 * non-whitespace input, and U+0000, which cannot be represented safely by
 * this parser's NUL-terminated string API. */
static inline jval *json_parse_exact(const char *text, char **arena_out) {
    if (!text) return NULL;
    jparser p = { text, NULL, 0, 0, 0, 1, 0 };
    jval *v = j_parse_val(&p);
    j_ws(&p);
    if (p.error || *p.s) {
        json_free(v);
        v = NULL;
    }
    if (arena_out) *arena_out = p.arena; else free(p.arena);
    return v;
}

#endif
