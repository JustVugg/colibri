/* Freeze production GLM DATA/ECHO bytes before sharing their transport. */
#include <stdio.h>
#include <assert.h>
#include <limits.h>
static FILE *glm_wire, *glm_input;
#undef stdin
#define stdin glm_input
#undef stdout
#define stdout glm_wire
#define printf(...) fprintf(glm_wire, __VA_ARGS__)
#define putchar(c) fputc((c), glm_wire)
#define main coli_glm_main_unused
#include "../colibri.c"
#undef main
#undef putchar
#undef printf
#undef stdout
#undef stdin

static FILE *wire_file(void) {
    FILE *file = tmpfile();
    assert(file);
#ifdef _WIN32
    assert(_setmode(_fileno(file), _O_BINARY) != -1);
#endif
    return file;
}

/* Exercise the real admission path without loading weights. Busy/duplicate
 * requests must consume both prompt and grammar before the next control frame. */
static void intake_case(const void *bytes, size_t size, int first_result,
                        const char *expected) {
    glm_input = wire_file();
    glm_wire = wire_file();
    assert(fwrite(bytes, 1, size, glm_input) == size);
    rewind(glm_input);
    ServeReq requests[2] = {{.active=1, .id=40}, {0}};
    assert(mux_submit(NULL, NULL, NULL, requests, NULL, 2, 16, 0) == first_result);
    if (first_result == 0) {
        assert(mux_submit(NULL, NULL, NULL, requests, NULL, 2, 16, 0) == 0);
        assert(mux_submit(NULL, NULL, NULL, requests, NULL, 2, 16, 0) == -1);
    }
    assert(requests[0].active && requests[0].id == 40 && !requests[1].active);
    rewind(glm_wire);
    char actual[256];
    size_t count = fread(actual, 1, sizeof(actual), glm_wire);
    assert(count == strlen(expected) && !memcmp(actual, expected, count));
    fclose(glm_input);
    fclose(glm_wire);
}

#define INTAKE(wire, result, expected) intake_case(wire, sizeof(wire)-1, result, expected)
static void test_intake(void) {
    INTAKE("SUBMIT 7 0 3 1 0 1\na\nb\nSTOP 99\n", 0,
           "ERROR 7 SLOT_BUSY\nERROR 99 NOT_FOUND\n");
    INTAKE("SUBMIT 8 0 3 1 0 1 4\nabcx\nyz\nCANCEL 99\n", 0,
           "ERROR 8 SLOT_BUSY\nERROR 99 NOT_FOUND\n");
    INTAKE("SUBMIT 18446744073709551615 0 3 0 0 1 4 pin=1 logprobs=2 ids=1\n1 2rule\nSTOP 99\n", 0,
           "ERROR 18446744073709551615 SLOT_BUSY\nERROR 99 NOT_FOUND\n");
    INTAKE("SUBMIT 40 1 0 1 0 1\n\nCANCEL 99\n", 0,
           "ERROR 40 DUPLICATE_ID\nERROR 99 NOT_FOUND\n");
    INTAKE("SUBMIT 9 2 0 1 0 1\n\nSTOP 99\n", 0,
           "ERROR 9 BAD_REQUEST\nERROR 99 NOT_FOUND\n");
    INTAKE("SUBMIT 10 0 3 1 0 1\na\0b\nSTOP 99\n", 0,
           "ERROR 10 BAD_REQUEST\nERROR 99 NOT_FOUND\n");
    INTAKE("SUBMIT 11 0 3 1 0 1 0 unknown=1\nabc\nSTOP 99\n", 0,
           "ERROR 11 BAD_REQUEST\nERROR 99 NOT_FOUND\n");
    INTAKE("STOP 0\nCANCEL 99\n", 0,
           "ERROR 0 BAD_REQUEST\nERROR 99 NOT_FOUND\n");
    INTAKE("CANCEL 40 extra\nSTOP 99\n", 0,
           "ERROR 0 BAD_REQUEST\nERROR 99 NOT_FOUND\n");
    INTAKE("SUBMIT 12 0 3 1 0 1\nab", -1, "");
    INTAKE("SUBMIT 12 0 3 1 0 1 4\nabcxy", -1, "");
    INTAKE("SUBMIT 12 0 3 1 0 1\nabc", -1, "ERROR 12 BAD_FRAME\n");
    INTAKE("SUBMIT 12 0 3 1 0 1\nabc\r\n", -1, "ERROR 12 BAD_FRAME\n");
}

int main(void) {
    glm_wire = tmpfile();
    assert(glm_wire);
#ifdef _WIN32
    assert(_setmode(_fileno(glm_wire), _O_BINARY) != -1);
#endif
    char *tokens[] = {"A\n", "\xc3\xa9", ""};
    int added[] = {1, 1, 1};
    Tok tokenizer = {0};
    tokenizer.id2str = tokens;
    tokenizer.id_added = added;
    tokenizer.n_ids = 3;
    float logits[] = {0, 0};
    mux_data(&tokenizer, 0, 0, NULL, 0, 0);
    mux_data(&tokenizer, ULLONG_MAX, 1, NULL, 0, 0);
    mux_data(&tokenizer, 9, 2, NULL, 0, 0);
    mux_data(&tokenizer, 10, 0, logits, 2, 1);
    mux_echo(&tokenizer, 11, 0, 0, NULL, 2, 1);
    mux_echo(&tokenizer, 11, 1, 1, logits, 2, 1);
    static const char expected[] =
        "DATA 0 2\nA\n\n"
        "DATA 18446744073709551615 2\n\xc3\xa9\n"
        "DATA 9 0\n\n"
        "DATA 10 2 -0.693147 1 0 -0.693147\nA\n\n"
        "ECHO 11 2 0 nan 0\nA\n\n"
        "ECHO 11 2 1 -0.693147 1 0 -0.693147\n\xc3\xa9\n";
    assert(fflush(glm_wire) == 0);
    rewind(glm_wire);
    char actual[sizeof(expected) + 1];
    size_t count = fread(actual, 1, sizeof(actual), glm_wire);
    assert(count == sizeof(expected) - 1);
    assert(memcmp(actual, expected, count) == 0);
    fclose(glm_wire);
    test_intake();
    puts("test_glm_serve_framing: ok");
    return 0;
}
