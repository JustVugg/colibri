/* schema_gbnf.h — JSON-Schema -> GBNF compiler for the grammar-forced draft source (#48/#70).
 *
 * Compiles a practical subset of JSON Schema into the byte-level GBNF subset that
 * grammar.h parses, so structured-output requests (OpenAI `response_format:
 * {"type":"json_schema"}` and the SCHEMA= env) get grammar-forced drafts without
 * hand-writing GBNF.
 *
 * Safety model: the grammar is a DRAFT SOURCE, never a sampling constraint (see
 * grammar.h). A schema compiled too strictly (or a model that deviates) only costs
 * draft acceptance — output is unchanged. So the compiler can be strict and compact
 * (no whitespace between tokens, exact key order): strictness maximizes forced-span
 * length on conforming outputs and cannot corrupt non-conforming ones.
 *
 * Supported subset:
 *   type: object   + properties (+ required)     — properties in declared order;
 *                    if `required` is present it must list every property (OpenAI
 *                    structured-output "strict" semantics); a proper subset -> fail.
 *   type: string   (+ enum of strings, const)
 *   type: number | integer | boolean | null
 *   type: array    + items (+ minItems 0|1)
 *   enum / const   also allowed at value level with numbers
 *   nesting to SGB_MAX_DEPTH; annotation keys ($schema, title, description,
 *   default, examples, additionalProperties) are ignored.
 * Anything else (anyOf/oneOf/$ref/pattern/format/minimum/maximum...) -> returns NULL and
 * the caller falls back to running without a grammar. Fail-closed, never fatal.
 */
#ifndef SCHEMA_GBNF_H
#define SCHEMA_GBNF_H
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "json.h"

#define SGB_MAX_DEPTH 32

typedef struct {
    char  *s; size_t len, cap;      /* output GBNF text */
    int    nrule;                   /* next composite rule id */
    int    use_str, use_num, use_int;  /* shared terminal rules actually referenced */
    char   err[160];
    int    fail;
} SgbCtx;

static void sgb_put(SgbCtx *C, const char *t){
    size_t n = strlen(t);
    if (C->len + n + 1 > C->cap){
        size_t nc = C->cap ? C->cap * 2 : 1024;
        while (nc < C->len + n + 1) nc *= 2;
        char *ns = (char *)realloc(C->s, nc);
        if (!ns){ C->fail = 1; return; }
        C->s = ns; C->cap = nc;
    }
    memcpy(C->s + C->len, t, n); C->len += n; C->s[C->len] = 0;
}

static void sgb_fail(SgbCtx *C, const char *what){
    if (!C->err[0]) snprintf(C->err, sizeof C->err, "unsupported schema: %s", what);
    C->fail = 1;
}

/* emit a JSON string VALUE (with quotes) as a GBNF literal: "..." with the JSON
 * text embedded, escaping for the GBNF literal syntax (\" \\ \xHH). */
static void sgb_put_json_string_lit(SgbCtx *C, const char *raw){
    sgb_put(C, "\"\\\"");                       /* GBNF literal opening: "\"  */
    char b[8];
    for (const unsigned char *p = (const unsigned char *)raw; *p; p++){
        unsigned char c = *p;
        if (c == '"')       sgb_put(C, "\\\\\\\"");   /* JSON \"  inside GBNF literal */
        else if (c == '\\') sgb_put(C, "\\\\\\\\");
        else if (c < 0x20){ snprintf(b, sizeof b, "\\x%02x", c); sgb_put(C, b); } /* raw ctl byte (invalid JSON anyway) */
        else if (c == 0x7f) sgb_put(C, "\\x7f");
        else { b[0] = (char)c; b[1] = 0; sgb_put(C, b); }
    }
    sgb_put(C, "\\\"\"");
}

/* emit a number the way a model would print it: shortest round-trip via %g */
static void sgb_put_number_lit(SgbCtx *C, double d){
    char b[64]; snprintf(b, sizeof b, "\"%.17g\"", d);
    /* trim %.17g noise for integers */
    if (d == (double)(long long)d && d < 1e15 && d > -1e15)
        snprintf(b, sizeof b, "\"%lld\"", (long long)d);
    sgb_put(C, b);
}

static int sgb_is_annotation(const char *k){
    return !strcmp(k,"$schema") || !strcmp(k,"title") || !strcmp(k,"description")
        || !strcmp(k,"default") || !strcmp(k,"examples")
        || !strcmp(k,"additionalProperties");
}

/* forward */
static void sgb_value(SgbCtx *C, jval *sc, int depth);

static void sgb_enum(SgbCtx *C, jval *e){
    if (!e || e->t != J_ARR || e->len < 1){ sgb_fail(C, "empty enum"); return; }
    sgb_put(C, "( ");
    for (int i = 0; i < e->len && !C->fail; i++){
        if (i) sgb_put(C, " | ");
        jval *v = e->kids[i];
        if (v->t == J_STR)       sgb_put_json_string_lit(C, v->str);
        else if (v->t == J_NUM)  sgb_put_number_lit(C, v->num);
        else if (v->t == J_BOOL) sgb_put(C, v->boolean ? "\"true\"" : "\"false\"");
        else if (v->t == J_NULL) sgb_put(C, "\"null\"");
        else sgb_fail(C, "enum member type");
    }
    sgb_put(C, " )");
}

static void sgb_object(SgbCtx *C, jval *sc, int depth){
    jval *props = json_get(sc, "properties");
    jval *req   = json_get(sc, "required");
    if (!props || props->t != J_OBJ){ sgb_fail(C, "object without properties"); return; }
    if (props->len == 0){ sgb_put(C, "\"{\" jws \"}\""); return; }
    if (req){
        if (req->t != J_ARR){ sgb_fail(C, "required not an array"); return; }
        /* strict semantics: every property must be required (OpenAI structured
         * outputs contract). A proper subset would need optional-group emission
         * with ambiguous separators — out of v1 scope. */
        if (req->len != props->len){ sgb_fail(C, "required must list every property (strict)"); return; }
        for (int i = 0; i < props->len; i++){
            int found = 0;
            for (int j = 0; j < req->len; j++)
                if (req->kids[j]->t == J_STR && !strcmp(req->kids[j]->str, props->keys[i])) found = 1;
            if (!found){ sgb_fail(C, "property not in required (strict)"); return; }
        }
    }
    /* jws at every separator: whitespace tolerance is strictly acceptance-positive
     * for a DRAFT-source grammar — a compact-only grammar dies (desyncs) at the
     * first stray space and forfeits every span after it, while jws points merely
     * aren't forced themselves (two legal bytes) and the multi-byte spans around
     * them keep drafting. Measured on GLM-5.2 current main: the sloppy-JSON
     * continuation costs a compact grammar most of its spans. */
    sgb_put(C, "\"{\" jws ");
    for (int i = 0; i < props->len && !C->fail; i++){
        if (i) sgb_put(C, " \",\" jws ");
        sgb_put_json_string_lit(C, props->keys[i]);
        sgb_put(C, " jws \":\" jws ");
        sgb_value(C, props->kids[i], depth + 1);
        sgb_put(C, " jws");
    }
    sgb_put(C, " \"}\"");
}

static void sgb_array(SgbCtx *C, jval *sc, int depth){
    jval *items = json_get(sc, "items");
    jval *mi    = json_get(sc, "minItems");
    int min1 = mi && mi->t == J_NUM && mi->num >= 1;
    if (mi && mi->t == J_NUM && mi->num > 1){ sgb_fail(C, "minItems > 1"); return; }
    if (!items){ sgb_fail(C, "array without items"); return; }
    if (min1){
        sgb_put(C, "\"[\" jws ");
        sgb_value(C, items, depth + 1);
        sgb_put(C, " jws ( \",\" jws ");
        sgb_value(C, items, depth + 1);
        sgb_put(C, " jws )* \"]\"");
    } else {
        sgb_put(C, "\"[\" jws ( ");
        sgb_value(C, items, depth + 1);
        sgb_put(C, " jws ( \",\" jws ");
        sgb_value(C, items, depth + 1);
        sgb_put(C, " jws )* )? \"]\"");
    }
}

static void sgb_value(SgbCtx *C, jval *sc, int depth){
    if (C->fail) return;
    if (depth > SGB_MAX_DEPTH){ sgb_fail(C, "nesting too deep"); return; }
    if (!sc || sc->t != J_OBJ){ sgb_fail(C, "schema node not an object"); return; }

    /* reject unknown constraint keywords (fail-closed) */
    for (int i = 0; i < sc->len; i++){
        const char *k = sc->keys[i];
        if (strcmp(k,"type") && strcmp(k,"properties") && strcmp(k,"required")
            && strcmp(k,"items") && strcmp(k,"enum") && strcmp(k,"const")
            && strcmp(k,"minItems") && !sgb_is_annotation(k)){
            sgb_fail(C, k); return;
        }
    }

    jval *cst = json_get(sc, "const");
    if (cst){
        if (cst->t == J_STR)       sgb_put_json_string_lit(C, cst->str);
        else if (cst->t == J_NUM)  sgb_put_number_lit(C, cst->num);
        else if (cst->t == J_BOOL) sgb_put(C, cst->boolean ? "\"true\"" : "\"false\"");
        else if (cst->t == J_NULL) sgb_put(C, "\"null\"");
        else sgb_fail(C, "const type");
        return;
    }
    jval *en = json_get(sc, "enum");
    if (en){ sgb_enum(C, en); return; }

    jval *ty = json_get(sc, "type");
    if (!ty || ty->t != J_STR){ sgb_fail(C, "missing type"); return; }
    const char *t = ty->str;
    if      (!strcmp(t, "object"))  sgb_object(C, sc, depth);
    else if (!strcmp(t, "array"))   sgb_array(C, sc, depth);
    else if (!strcmp(t, "string")){ C->use_str = 1; sgb_put(C, "jstr"); }
    else if (!strcmp(t, "number")){ C->use_num = 1; sgb_put(C, "jnum"); }
    else if (!strcmp(t, "integer")){ C->use_int = 1; sgb_put(C, "jint"); }
    else if (!strcmp(t, "boolean")) sgb_put(C, "( \"true\" | \"false\" )");
    else if (!strcmp(t, "null"))    sgb_put(C, "\"null\"");
    else sgb_fail(C, t);
}

static void sgb_free_jval(jval *v){
    if (!v) return;
    if (v->t == J_OBJ){
        for (int i = 0; i < v->len; i++){ free(v->keys[i]); sgb_free_jval(v->kids[i]); }
        free(v->keys); free(v->kids);
    } else if (v->t == J_ARR){
        for (int i = 0; i < v->len; i++) sgb_free_jval(v->kids[i]);
        free(v->kids);
    } else if (v->t == J_STR) free(v->str);
    free(v);
}

/* Compile a JSON-Schema string to GBNF. Returns a malloc'd GBNF text (caller
 * frees) or NULL with a message in err (if err != NULL). */
static char *schema_to_gbnf(const char *schema_json, char *err, int errsz){
    SgbCtx C; memset(&C, 0, sizeof C);
    jval *sc = json_parse(schema_json, NULL);
    if (!sc){ if (err) snprintf(err, errsz, "schema: json parse failed"); return NULL; }

    sgb_put(&C, "root ::= jws ");
    sgb_value(&C, sc, 0);
    sgb_put(&C, " jws\n");
    sgb_put(&C, "jws ::= ( \" \" | \"\\t\" | \"\\n\" | \"\\r\" )*\n");
    if (C.use_str)
        sgb_put(&C, "jstr ::= \"\\\"\" jchar* \"\\\"\"\n"
                    "jchar ::= [^\"\\\\\\x00-\\x1f] | \"\\\\\" ( [\"\\\\/bfnrt] | \"u\" jhex jhex jhex jhex )\n"
                    "jhex ::= [0-9a-fA-F]\n");
    if (C.use_num)
        sgb_put(&C, "jnum ::= \"-\"? ( \"0\" | [1-9] [0-9]* ) ( \".\" [0-9]+ )? ( ( \"e\" | \"E\" ) ( \"+\" | \"-\" )? [0-9]+ )?\n");
    if (C.use_int)
        sgb_put(&C, "jint ::= \"-\"? ( \"0\" | [1-9] [0-9]* )\n");

    sgb_free_jval(sc);
    if (C.fail || !C.s){
        if (err) snprintf(err, errsz, "%s", C.err[0] ? C.err : "schema: compile failed");
        free(C.s); return NULL;
    }
    return C.s;
}

#endif /* SCHEMA_GBNF_H */
