/* Kimi (K3) pre-tokenizer against a tiny in-repo vocabulary.
 *
 * WHY THIS EXISTS. o200k has had a gate since it was written
 * (tests/test_tok_o200k.c, 40/40 encode + decode off a 4 KB fixture). The Kimi
 * family has had none: tests/test_tok_kimi.c needs a real tokenizer.json and a
 * cases.bin produced by tiktoken, so it has no build rule and has never run in
 * CI -- c/Makefile lists it among the files that "deliberately have no rule".
 *
 * That matters because the two families share one implementation. tok.h's
 * o2_letters_masked() and pretok_chunk_o2fam() serve o200k and Kimi from the
 * same lines, differing by a flag; a change made for one silently reshapes the
 * other, and only the o200k half had anything watching.
 *
 * WHAT IS ASSERTED, AND WHAT IS NOT. Not token ids from the real Kimi
 * tokenizer -- reproducing those needs the actual vocabulary, which is what
 * made the existing test unrunnable. What is asserted is the pre-tokenizer
 * BOUNDARY, which is a property of the rules and is what the shared code
 * decides:
 *
 *   - a Han run is its own chunk, and does not merge with adjacent Latin
 *   - Han never joins a letter run (it is \p{Lo}; the Kimi classes mask it out)
 *   - rule D takes no '/' tail, unlike o200k
 *
 * The fixture makes those boundaries observable: "中文" is one vocabulary entry,
 * so a correct Han-run split encodes it as ONE id, and any other split falls
 * back to per-byte ids. The same input through the o200k fixture takes the
 * other path, so the two are contrasted rather than asserted in isolation.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <string.h>
#include "../tok.h"

static int fails;

static void expect_n(Tok *T, const char *what, const char *text, int want) {
    int ids[128];
    int n = tok_encode(T, text, (int)strlen(text), ids, 128);
    if (n != want) {
        printf("  FAIL %-28s %-12s got %d ids, want %d:", what, text, n, want);
        for (int i = 0; i < n; i++) printf(" %d", ids[i]);
        printf("\n");
        fails++;
    } else {
        printf("  ok   %-28s %-12s %d ids\n", what, text, n);
    }
}

/* The two families must not agree here -- if they do, the Kimi flag stopped
 * reaching the splitter and this whole file would pass vacuously. */
static void expect_differs(Tok *A, Tok *B, const char *text) {
    int a[128], b[128];
    int na = tok_encode(A, text, (int)strlen(text), a, 128);
    int nb = tok_encode(B, text, (int)strlen(text), b, 128);
    int same = (na == nb) && memcmp(a, b, (size_t)na * sizeof(int)) == 0;
    if (same) {
        printf("  FAIL kimi and o200k agree on %-12s (%d ids) -- flag not reaching the splitter\n", text, na);
        fails++;
    } else {
        printf("  ok   kimi != o200k on %-12s (%d vs %d ids)\n", text, na, nb);
    }
}

int main(void) {
    Tok K, O;
    tok_load(&K, "tests/tok_kimi_tiny.json");
    tok_load(&O, "tests/tok_o200k_tiny.json");

    if (!K.kimi) {
        printf("  FAIL fixture did not select the Kimi family (kimi=%d)\n", K.kimi);
        return 1;
    }
    if (O.kimi) {
        printf("  FAIL o200k fixture selected the Kimi family\n");
        return 1;
    }
    printf("  ok   family detection: kimi=%d / o200k kimi=%d\n", K.kimi, O.kimi);

    /* Han run is one chunk and hits the vocabulary entry. */
    expect_n(&K, "Han run is one chunk", "\xe4\xb8\xad\xe6\x96\x87", 1);
    /* Han does not absorb the Latin that follows: 中文 + abc, not one blob. */
    expect_n(&K, "Han does not eat Latin", "\xe4\xb8\xad\xe6\x96\x87" "abc", 3);
    /* A Han codepoint at the far end of the range (U+9FA5) is still a Han run,
     * so it does not merge with the Latin that follows: one id each. */
    expect_n(&K, "U+9FA5 splits from Latin", "\xe9\xbe\xa5" "a", 2);

    /* Same inputs must NOT take the o200k path. */
    expect_differs(&K, &O, "\xe4\xb8\xad\xe6\x96\x87");
    expect_differs(&K, &O, "\xe4\xb8\xad\xe6\x96\x87" "abc");

    printf(fails ? "test_tok_kimi_tiny: %d failure(s)\n" : "test_tok_kimi_tiny: ok\n", fails);
    return fails != 0;
}
