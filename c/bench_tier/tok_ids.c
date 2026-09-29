/* Tokenize a text prompt into OLMoE token ids for the ref.json fixture.
 * Usage: ./tok_ids <tokenizer.json> "<text>"  -> prints space-separated ids.
 * Measurement/fixture tool only; does not touch the engine. */
#include <stdio.h>
#include <string.h>
#include "../tok.h"

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <tokenizer.json> <text>\n", argv[0]); return 2; }
    Tok T;
    memset(&T, 0, sizeof(T));
    tok_load(&T, argv[1]);
    int ids[32768];
    int n = tok_encode(&T, argv[2], (int)strlen(argv[2]), ids, 32768);
    if (n <= 0) { fprintf(stderr, "encode failed: %d\n", n); return 1; }
    for (int i = 0; i < n; i++) printf("%d%s", ids[i], i + 1 < n ? " " : "\n");
    tok_free(&T);
    return 0;
}
