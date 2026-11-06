/* Freeze production GLM DATA/ECHO bytes before sharing their transport. */
#include <stdio.h>
#include <assert.h>
#include <limits.h>
static FILE *glm_wire;
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
    puts("test_glm_serve_framing: ok");
    return 0;
}
