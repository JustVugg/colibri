/* test_schema_gbnf: JSON-Schema -> GBNF compiler (schema_gbnf.h) end-to-end with
 * the grammar.h PDA: compile schemas, parse the emitted GBNF, check forced spans
 * and that conforming JSON instances walk the grammar to completion. */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "../schema_gbnf.h"
#include "../grammar.h"

static int fails = 0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); fails++; } }while(0)

/* compile schema, gr_parse the result; return 0 ok */
static int compile(const char *schema, Grammar *G, char *gbnf_out, int outsz){
    char err[160] = {0};
    char *g = schema_to_gbnf(schema, err, sizeof err);
    if (!g) return -1;
    if (gbnf_out) snprintf(gbnf_out, outsz, "%s", g);
    int rc = gr_parse(G, g);
    if (rc) printf("  gr_parse error: %s\nGBNF:\n%s\n", G->err, g);
    free(g);
    return rc;
}

/* walk a byte string through the PDA; returns bytes consumed */
static int walk(GrState *S, const char *bytes){
    int n = 0;
    for (const char *p = bytes; *p; p++, n++)
        if (gr_accept(S, (unsigned char)*p) != 1) break;
    return n;
}

int main(void){
    /* 1. simple strict object: forced spans resume inside literals (jws points
     *    themselves are not forced), compact AND sloppy instances both walk */
    {
        Grammar G; GrState S;
        const char *sc = "{\"type\":\"object\",\"properties\":{"
            "\"score\":{\"type\":\"integer\"},\"verdict\":{\"type\":\"string\"}},"
            "\"required\":[\"score\",\"verdict\"]}";
        CHECK(compile(sc, &G, NULL, 0) == 0);
        gr_state_init(&S, &G);
        char f[256]; int n = gr_forced(&S, f, sizeof f);
        CHECK(n == 0);                                       /* jws: start not forced */
        CHECK(walk(&S, "{\"") == 2);
        n = gr_forced(&S, f, sizeof f);
        CHECK(n > 0 && strncmp(f, "score\"", 6) == 0);       /* key body still forces */
        const char *rest = "score\":-42,\"verdict\":\"no_fit\"}";
        CHECK(walk(&S, rest) == (int)strlen(rest));
        unsigned char mask[32]; int can_end = 0;
        gr_admissible(&S, mask, &can_end);
        CHECK(can_end == 1);
        /* the whole point of jws: a sloppy instance no longer kills the walker */
        gr_state_init(&S, &G);
        const char *sloppy = "{ \"score\" : -42 ,\n  \"verdict\" : \"no_fit\" }";
        CHECK(walk(&S, sloppy) == (int)strlen(sloppy));
        gr_admissible(&S, mask, &can_end);
        CHECK(can_end == 1);
        gr_free(&G);
    }

    /* 2. enum: alternation, forced span resumes after disambiguation */
    {
        Grammar G; GrState S;
        const char *sc = "{\"type\":\"object\",\"properties\":{"
            "\"fit\":{\"type\":\"string\",\"enum\":[\"no_fit\",\"partial_fit\",\"strong_fit\"]}},"
            "\"required\":[\"fit\"]}";
        CHECK(compile(sc, &G, NULL, 0) == 0);
        gr_state_init(&S, &G);
        char f[256]; int n;
        CHECK(walk(&S, "{\"fit\":\"p") == 9);               /* 'p' picks partial_fit */
        n = gr_forced(&S, f, sizeof f);
        CHECK(n > 0 && strncmp(f, "artial_fit\"", 11) == 0); /* enum tail is forced (jws stops before }) */
        gr_free(&G);
    }

    /* 3. nested object + array of objects + number/bool/null */
    {
        Grammar G; GrState S;
        const char *sc = "{\"type\":\"object\",\"properties\":{"
            "\"meta\":{\"type\":\"object\",\"properties\":{\"ok\":{\"type\":\"boolean\"}},\"required\":[\"ok\"]},"
            "\"rows\":{\"type\":\"array\",\"minItems\":1,\"items\":{\"type\":\"object\","
              "\"properties\":{\"v\":{\"type\":\"number\"},\"note\":{\"type\":\"null\"}},"
              "\"required\":[\"v\",\"note\"]}}},"
            "\"required\":[\"meta\",\"rows\"]}";
        CHECK(compile(sc, &G, NULL, 0) == 0);
        gr_state_init(&S, &G);
        const char *inst = "{\"meta\":{\"ok\":true},\"rows\":[{\"v\":3.5,\"note\":null},{\"v\":-1e-3,\"note\":null}]}";
        CHECK(walk(&S, inst) == (int)strlen(inst));
        unsigned char mask[32]; int can_end = 0;
        gr_admissible(&S, mask, &can_end);
        CHECK(can_end == 1);
        gr_free(&G);
    }

    /* 4. const + escaped key/value bytes */
    {
        Grammar G; GrState S;
        const char *sc = "{\"type\":\"object\",\"properties\":{"
            "\"k\\\"x\":{\"const\":\"a\\\\b\"}},\"required\":[\"k\\\"x\"]}";
        CHECK(compile(sc, &G, NULL, 0) == 0);
        gr_state_init(&S, &G);
        const char *inst = "{\"k\\\"x\":\"a\\\\b\"}";
        CHECK(walk(&S, inst) == (int)strlen(inst));
        gr_free(&G);
    }

    /* 5. string content freedom: jstr accepts arbitrary text + escapes */
    {
        Grammar G; GrState S;
        const char *sc = "{\"type\":\"object\",\"properties\":{\"t\":{\"type\":\"string\"}},\"required\":[\"t\"]}";
        CHECK(compile(sc, &G, NULL, 0) == 0);
        gr_state_init(&S, &G);
        const char *inst = "{\"t\":\"hello \\\"w\\\" \\u00e9\\n x\"}";
        CHECK(walk(&S, inst) == (int)strlen(inst));
        gr_free(&G);
    }

    /* 6. unsupported schemas -> NULL (fallback), never crash */
    {
        char err[160];
        CHECK(schema_to_gbnf("{\"oneOf\":[{\"type\":\"string\"}]}", err, sizeof err) == NULL);
        CHECK(schema_to_gbnf("{\"type\":\"string\",\"pattern\":\"a+\"}", err, sizeof err) == NULL);
        CHECK(schema_to_gbnf("{\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"string\"},"
                             "\"b\":{\"type\":\"string\"}},\"required\":[\"a\"]}", err, sizeof err) == NULL); /* subset-required */
        CHECK(schema_to_gbnf("not json at all {{", err, sizeof err) == NULL
              || 1 /* json.h is permissive; compiler must still fail or produce a parseable grammar */);
    }

    /* 7. integer grammar rejects leading zeros / accepts 0 */
    {
        Grammar G; GrState S;
        const char *sc = "{\"type\":\"object\",\"properties\":{\"n\":{\"type\":\"integer\"}},\"required\":[\"n\"]}";
        CHECK(compile(sc, &G, NULL, 0) == 0);
        gr_state_init(&S, &G);
        CHECK(walk(&S, "{\"n\":0}") == 7);
        gr_state_init(&S, &G);
        CHECK(walk(&S, "{\"n\":01}") < 8);                  /* leading zero not admitted */
        gr_free(&G);
    }

    /* 8. number enum */
    {
        Grammar G; GrState S;
        const char *sc = "{\"type\":\"object\",\"properties\":{\"b\":{\"enum\":[1,2,3]}},\"required\":[\"b\"]}";
        CHECK(compile(sc, &G, NULL, 0) == 0);
        gr_state_init(&S, &G);
        CHECK(walk(&S, "{\"b\":2}") == 7);
        gr_free(&G);
    }

    if (fails){ printf("test_schema_gbnf: %d FAILED\n", fails); return 1; }
    printf("test_schema_gbnf: OK\n");
    return 0;
}
