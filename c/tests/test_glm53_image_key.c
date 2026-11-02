/* test_glm53_image_key — the KV reuse decision when a request carries an image.
 *
 * Two requests can have identical token ids and different pictures: the
 * gateway expands every image into the same run of `image_token_id`, and the
 * pixels arrive in their own frame. If the slot compared ids alone, the second
 * request would answer from the attention state built on the first picture —
 * no crash, a plausible reply about the wrong photo. So the state-identity
 * rule gets its own cases here: same placeholders, different bytes → nothing
 * shared; same bytes → everything shared, and the tower outputs the prefix
 * already holds are counted so the caller can skip encoding them.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../kv_image_key.h"

static int failures;
#define CHECK(cond, ...) do { if (!(cond)) { \
    fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
    fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); failures++; } } while (0)

/* The reuse rule glm53.c applies: the cached prefix is taken whole or not at
 * all, and there must be at least one new position to prefill. */
static int reuse(const uint64_t *have, int cached, const uint64_t *want, int n) {
    if (cached <= 0 || cached >= n) return 0;
    return kv_keys_shared(have, cached, want, n) >= cached ? cached : 0;
}

int main(void) {
    enum { IMG = 5, VOCAB = 200000 };
    const float a[8] = { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f };
    float b[8]; memcpy(b, a, sizeof(a)); b[3] = 0.41f;   /* one patch value off */

    /* ---- the hash names content, not the request ---- */
    const uint64_t ha = kv_image_hash(a, sizeof(a), 4, 4);
    CHECK(ha == kv_image_hash(a, sizeof(a), 4, 4), "same bytes, same grid: same hash");
    CHECK(ha != kv_image_hash(b, sizeof(b), 4, 4), "one value off: another image");
    CHECK(ha != kv_image_hash(a, sizeof(a), 2, 8), "same bytes at another grid: another image");
    CHECK(ha != kv_image_hash(a, sizeof(a) - 4, 4, 4), "a truncated payload is another image");

    /* ---- keys never collide with token ids ---- */
    const uint64_t ka = kv_image_key(ha), kb = kv_image_key(kv_image_hash(b, sizeof(b), 4, 4));
    CHECK(kv_key_is_image(ka) && kv_key_is_image(kb), "image keys carry the image bit");
    CHECK(kv_key_is_image(KV_IMAGE_KEY_NONE), "the frameless placeholder is still an image position");
    CHECK(ka != KV_IMAGE_KEY_NONE, "a real image is not the frameless placeholder");
    {
        const int ids[3] = { 0, VOCAB - 1, 0x7fffffff };
        uint64_t keys[3];
        kv_keys_build(keys, ids, 3, IMG, ka);
        for (int t = 0; t < 3; t++) {
            CHECK(!kv_key_is_image(keys[t]), "token id %d must not read as an image", ids[t]);
            CHECK(keys[t] == (uint64_t)ids[t], "token id %d keeps its id", ids[t]);
        }
        const int neg[1] = { -1 };
        kv_keys_build(keys, neg, 1, IMG, ka);
        CHECK(!kv_key_is_image(keys[0]), "a negative id must not read as an image");
        kv_keys_build(keys, ids, 3, -1, ka);         /* checkpoint without an image token */
        CHECK(keys[0] == 0 && keys[2] == 0x7fffffff, "no image token: nothing is substituted");
    }

    /* ---- T10: identical placeholders, different content → no shared KV ---- */
    const int turn1[]  = { 7, IMG, IMG, IMG, 9 };           /* what the slot holds */
    const int turn2[]  = { 7, IMG, IMG, IMG, 9, 11 };       /* extends it by one token */
    const int cached = 5;
    uint64_t have[5], want[6];
    kv_keys_build(have, turn1, cached, IMG, ka);           /* built on picture A */

    kv_keys_build(want, turn2, 6, IMG, kb);                /* picture B, same ids */
    CHECK(kv_keys_shared(have, cached, want, 6) == 1,
          "different image: agreement stops at the first placeholder");
    CHECK(reuse(have, cached, want, 6) == 0, "different image: no reuse");

    kv_keys_build(want, turn2, 6, IMG, ka);                /* picture A again */
    CHECK(kv_keys_shared(have, cached, want, 6) == cached, "same image: whole prefix agrees");
    CHECK(reuse(have, cached, want, 6) == cached, "same image: the prefix is reused");
    CHECK(kv_keys_images(want, cached) == 3,
          "the reused prefix already holds all 3 tower outputs");
    CHECK(kv_keys_images(want, 1) == 0, "no image position before the span");
    CHECK(kv_keys_images(want, 2) == 1, "a prefix ending mid-span holds one");

    kv_keys_build(want, turn2, 6, IMG, KV_IMAGE_KEY_NONE);  /* placeholders, no frame */
    CHECK(reuse(have, cached, want, 6) == 0, "no frame: not the picture the slot holds");

    /* the slot built on frameless placeholders matches only the same */
    kv_keys_build(have, turn1, cached, IMG, KV_IMAGE_KEY_NONE);
    CHECK(reuse(have, cached, want, 6) == cached, "frameless then frameless: same input");
    kv_keys_build(want, turn2, 6, IMG, ka);
    CHECK(reuse(have, cached, want, 6) == 0, "frameless then picture A: not the same input");

    /* ---- the text rules did not move ---- */
    const int text1[] = { 7, 8, 9 }, text2[] = { 7, 8, 9, 10 }, other[] = { 7, 99, 9, 10 };
    uint64_t t1[3], t2[4], t3[4];
    kv_keys_build(t1, text1, 3, IMG, ka);
    kv_keys_build(t2, text2, 4, IMG, ka);
    kv_keys_build(t3, other, 4, IMG, ka);
    CHECK(reuse(t1, 3, t2, 4) == 3, "an extending text prompt reuses all 3");
    CHECK(reuse(t1, 3, t3, 4) == 0, "a diverging text prompt reuses nothing");
    CHECK(reuse(t1, 3, t1, 3) == 0, "equal length: nothing new to prefill, no reuse");
    CHECK(reuse(t1, 0, t2, 4) == 0, "an empty slot reuses nothing");

    if (failures) { fprintf(stderr, "%d failure(s)\n", failures); return 1; }
    puts("test_glm53_image_key: ok");
    return 0;
}
