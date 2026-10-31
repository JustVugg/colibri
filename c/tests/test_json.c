#include <math.h>
#include <stdio.h>
#include <string.h>

#include "../json.h"

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #condition); \
        return 1; \
    } \
} while (0)

int main(void) {
    jval *root = json_parse(
        "{\"name\":\"Colibri\\nCPU\",\"enabled\":true,\"empty\":null,"
        "\"values\":[1,-2.5,3e2],\"unicode\":\"\\u03bb \\uD83D\\uDE80\"}",
        NULL
    );

    CHECK(root && root->t == J_OBJ);
    CHECK(strcmp(json_get(root, "name")->str, "Colibri\nCPU") == 0);
    CHECK(json_get(root, "enabled")->boolean == 1);
    CHECK(json_get(root, "empty")->t == J_NULL);
    CHECK(json_get(root, "missing") == NULL);

    jval *values = json_get(root, "values");
    CHECK(values->t == J_ARR && values->len == 3);
    CHECK(values->kids[0]->num == 1.0);
    CHECK(values->kids[1]->num == -2.5);
    CHECK(values->kids[2]->num == 300.0);
    CHECK(strcmp(json_get(root, "unicode")->str, "λ 🚀") == 0);
    json_free(root);

    /* An unpaired surrogate is not a Unicode scalar. Checked parse rejects it;
     * a paired surrogate (the rocket above) still decodes. Legacy stays permissive. */
    CHECK(json_parse_checked("\"\\uD800\"") == NULL);
    CHECK(json_parse_checked("\"\\uDE80\"") == NULL);
    CHECK(json_parse_checked("\"\\uD800\\u0041\"") == NULL);
    root = json_parse_checked("\"\\uD83D\\uDE80\"");
    CHECK(root && strcmp(root->str, "🚀") == 0);
    json_free(root);
    root = json_parse("\"\\uD800\"", NULL);
    CHECK(root && root->t == J_STR);
    json_free(root);

    /* Checked key lookup must not alias a name through a decoded NUL. The
     * legacy API still returns its original permissive representation. */
    const char *nul_key="{\"x\\u0000ignored\":1}";
    CHECK(json_parse_checked(nul_key) == NULL);
    root=json_parse(nul_key,NULL);
    CHECK(root && json_get(root,"x") && json_get(root,"x")->num == 1.0);
    json_free(root);

    /* Legal escapes, including NUL in a string value, remain valid. */
    root=json_parse_checked("{\"\\u0078\":\"a\\u0000b\\f\\u000b\"}");
    CHECK(root && json_get(root,"x"));
    CHECK(memcmp(json_get(root,"x")->str,"a\0b\f\v",5) == 0);
    json_free(root);

    const char whitespace[]=" \t\r\n\f\v";
    const char *positions[]={
        "%c{\"x\":[1]}", "{%c\"x\":[1]}", "{\"x\"%c:[1]}",
        "{\"x\":%c[1]}", "{\"x\":[%c1]}", "{\"x\":[1%c]}",
        "{\"x\":[1]%c}", "{\"x\":[1]}%c",
    };
    for (size_t i=0; i<sizeof(whitespace)-1; i++) {
        for (size_t j=0; j<sizeof(positions)/sizeof(positions[0]); j++) {
            char input[64];
            snprintf(input,sizeof(input),positions[j],whitespace[i]);
            root=json_parse_checked(input);
            CHECK((root != NULL) == (i < 4));
            json_free(root);
            root=json_parse(input,NULL);
            values=json_get(root,"x");
            CHECK(values && values->t == J_ARR && values->len == 1);
            CHECK(values->kids[0]->num == 1.0);
            json_free(root);
        }
    }

    /* The qpack reader goes through json_parse_checked: exact grammar, no
     * trailing input, no NUL inside an object key. */
    root = json_parse_checked(" {\"exact\":[0,-2.5,3e2]} \n");
    CHECK(root && root->t == J_OBJ);
    json_free(root);

    static const char *malformed[] = {
        "{\"a\" 1}",
        "{\"a\":\"unterminated}",
        "{\"a\":1",
        "{\"a\":1} trailing",
        "{\"a\":1,}",
        "[1,]",
        "{\"a\":1e}",
        "{\"magic\\u0000suffix\":\"QPACK\"}",
        "{\"a\":1}\v"
    };
    for (size_t i = 0; i < sizeof(malformed) / sizeof(malformed[0]); i++)
        CHECK(json_parse_checked(malformed[i]) == NULL);

    /* Overflow to infinity is not a value a checked reader can store. */
    CHECK(json_parse_checked("1e9999") == NULL);
    CHECK(json_parse_checked("{\"a\":-1e9999}") == NULL);
    root = json_parse("1e9999", NULL);
    CHECK(root && !isfinite(root->num));
    json_free(root);
    root = json_parse_checked("1e308");
    CHECK(root && isfinite(root->num));
    json_free(root);
    root = json_parse_checked("3e2");
    CHECK(root && root->num == 300.0);
    json_free(root);


    puts("json tests: ok");
    return 0;
}
