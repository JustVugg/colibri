/* A serving payload is byte-counted, not NUL-terminated text: serve_read_req
 * reads exactly plen bytes into malloc(plen+1) and serve_one hands them to
 * encode_text as raw prompt text.  A prompt whose tail is a truncated multibyte
 * sequence therefore reaches the pre-tokenizer with a lead byte whose
 * continuation bytes are not there.
 *
 * qwen36 and qwen38 carry the same pre-tokenizer.  qwen38's copy treats that
 * byte as one invalid unit and is gated in tests/test_qwen38_tokenizer.c;
 * qwen36's copy read the continuation bytes anyway and reported the lead byte's
 * full length, so pretok_end returned a piece end past the prompt.  These are
 * the sibling's assertions, run against this engine's copy. */
#define _GNU_SOURCE
#define main qwen36_main_unused
#include "../qwen36.c"
#undef main

#define CHECK(condition) do {                                               \
    if (!(condition)) {                                                     \
        fprintf(stderr, "%s:%d: check failed: %s\n",                        \
                __FILE__, __LINE__, #condition);                            \
        return 1;                                                           \
    }                                                                       \
} while (0)

/* the buffer serve_read_req builds: exactly n bytes plus the NUL */
static char *payload(const char *bytes, int n) {
    char *p = (char *)malloc((size_t)n + 1);
    if (!p) return NULL;
    memcpy(p, bytes, (size_t)n);
    p[n] = 0;
    return p;
}

int main(void) {
    /* the gate tests/test_qwen38_tokenizer.c already applies to the sibling */
    const char lead3[] = { (char)0xe2, 0 };
    int adv = 0;
    CHECK(utf8_decode(lead3, 0, 1, &adv) == 0xe2);
    CHECK(adv == 1);
    CHECK(pretok_end(lead3, 0, 1) == 1);

    /* a truncated 2-byte lead: pretok_end must not walk past n */
    {
        char *p = payload("a\xc3", 2);
        CHECK(p != NULL);
        CHECK(pretok_end(p, 1, 2) == 2);
        free(p);
    }

    /* a truncated 4-byte lead: utf8_adv must report one byte, not four */
    {
        char *p = payload("hi \xf0\x9f", 5);
        CHECK(p != NULL);
        CHECK(utf8_adv(p, 3, 5) == 1);
        CHECK(pretok_end(p, 3, 5) <= 5);
        free(p);
    }

    /* an out-of-range index answers U+FFFD with advance 0, as qwen38 does */
    CHECK(utf8_decode(lead3, 1, 1, &adv) == 0xfffd);
    CHECK(adv == 0);

    /* well-formed input is untouched: the decode is still one whole character */
    {
        const char *ok = "\xe4\xbd\xa0";          /* U+4F60 */
        CHECK(utf8_decode(ok, 0, 3, &adv) == 0x4f60);
        CHECK(adv == 3);
        CHECK(utf8_adv(ok, 0, 3) == 3);
    }

    puts("test_qwen36_tok_truncated: ok");
    return 0;
}
