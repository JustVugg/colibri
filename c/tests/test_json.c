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

    puts("json tests: ok");
    return 0;
}
