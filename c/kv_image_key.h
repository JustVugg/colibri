/* kv_image_key.h — name an image span in a KV history by its content.
 *
 * A prefix-reuse record made of token ids cannot tell two images apart: the
 * gateway expands every image into the same run of `image_token_id`
 * placeholders, and the pixels travel beside the prompt in their own frame.
 * Two requests that differ only in the picture therefore have identical id
 * sequences, and an id-only comparison would reuse the attention state built
 * on the OTHER picture — an answer about the wrong photo, with no error.
 * glm53.c used to dodge this by throwing the cached prefix away whenever an
 * image was pending; kv_prefix.h's taint bit does the same for inkling's
 * audio. Both are safe and both pay a full prefill for the image that was
 * already there.
 *
 * THE IDEA. The position of an image token is keyed by what the tower was
 * given, not by the placeholder id: a 64-bit content hash of the patches and
 * their grid, with the top bit set so it can never equal a token id. Real
 * tokens keep their ids. Two histories then agree on an image position only
 * when the same bytes reached the tower, and the reuse decision stays the
 * same all-or-nothing comparison it always was. A placeholder that arrived
 * without a frame keys as "no content", which matches only another such
 * placeholder — the embedding table row is what both saw.
 *
 * The hash is FNV-1a over the grid and the patch bytes as the gateway sent
 * them (post-resize, pre-tower). Two different images meet on the same
 * 63-bit key with probability 2^-63 per pair.
 */
#ifndef KV_IMAGE_KEY_H
#define KV_IMAGE_KEY_H

#include <stddef.h>
#include <stdint.h>

#define KV_IMAGE_KEY_BIT (UINT64_C(1) << 63)

/* Content hash of one image as the tower receives it: the grid first, so the
 * same bytes at another geometry are another image, then the patches. */
static inline uint64_t kv_image_hash(const void *patches, size_t bytes,
                                     int grid_h, int grid_w) {
    uint64_t h = UINT64_C(0xcbf29ce484222325);
    const uint64_t prime = UINT64_C(0x100000001b3);
    const int dims[2] = { grid_h, grid_w };
    const unsigned char *p = (const unsigned char *)dims;
    for (size_t i = 0; i < sizeof(dims); i++) { h ^= p[i]; h *= prime; }
    p = (const unsigned char *)patches;
    for (size_t i = 0; i < bytes; i++) { h ^= p[i]; h *= prime; }
    return h;
}

/* The key an image position carries: above every token id by construction. */
static inline uint64_t kv_image_key(uint64_t hash) {
    return hash | KV_IMAGE_KEY_BIT;
}

/* Key for a placeholder the engine received no frame for. */
#define KV_IMAGE_KEY_NONE KV_IMAGE_KEY_BIT

static inline int kv_key_is_image(uint64_t key) {
    return (key & KV_IMAGE_KEY_BIT) != 0;
}

/* Build the keyed sequence: token ids as they are, image positions as
 * `image_key`. `image_token < 0` means the checkpoint has no image token and
 * nothing is substituted. Ids are widened unsigned so a stray negative id can
 * never carry the image bit. */
static inline void kv_keys_build(uint64_t *keys, const int *tokens, int n,
                                 int image_token, uint64_t image_key) {
    for (int t = 0; t < n; t++)
        keys[t] = (image_token >= 0 && tokens[t] == image_token)
                  ? image_key : (uint64_t)(uint32_t)tokens[t];
}

/* Leading positions on which two keyed sequences agree. */
static inline int kv_keys_shared(const uint64_t *have, int have_n,
                                 const uint64_t *want, int want_n) {
    int shared = 0;
    while (shared < have_n && shared < want_n && have[shared] == want[shared])
        shared++;
    return shared;
}

/* How many of the first n positions are image positions: the tower outputs a
 * reused prefix already holds, so the caller can skip that many embeddings. */
static inline int kv_keys_images(const uint64_t *keys, int n) {
    int count = 0;
    for (int t = 0; t < n; t++) count += kv_key_is_image(keys[t]);
    return count;
}

#endif /* KV_IMAGE_KEY_H */
