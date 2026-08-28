/* Regression gate for qwen36 OpenAI JSON string escaping. */
#define main qwen36_main_unused
#include "../qwen36.c"
#undef main

static int failures;

#define CHECK(cond, what) do { \
    if (!(cond)) { fprintf(stderr, "FAIL: %s\n", (what)); failures++; } \
    else printf("ok: %s\n", (what)); \
} while (0)

static void case_json_escapes(void) {
    static const unsigned char input[] = {
        'a', '"', 'b', '\\', 'c', '\n', '\r', '\t', '\b', '\f', 0x01
    };
    char out[128];
    int n = json_escape(input, (int)sizeof input, out, (int)sizeof out);
    const char *want = "a\\\"b\\\\c\\n\\r\\t\\b\\f\\u0001";

    CHECK(strcmp(out, want) == 0, "quotes, backslashes and controls use JSON escapes");
    CHECK(n == (int)strlen(want), "returned length matches escaped output");
}

static void case_plain_utf8_passes_through(void) {
    static const unsigned char input[] = { 'x', 0xE2, 0x82, 0xAC, 'y' };
    char out[32];
    int n = json_escape(input, (int)sizeof input, out, (int)sizeof out);

    CHECK(n == (int)sizeof input, "plain UTF-8 byte length is unchanged");
    CHECK(memcmp(out, input, sizeof input) == 0 && out[sizeof input] == '\0',
          "plain UTF-8 bytes pass through unchanged");
}

int main(void) {
    case_json_escapes();
    case_plain_utf8_passes_through();
    if (failures) {
        fprintf(stderr, "%d qwen36 JSON escape regression(s)\n", failures);
        return 1;
    }
    puts("qwen36 JSON escape regression: PASS");
    return 0;
}
