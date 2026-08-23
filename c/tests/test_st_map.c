#define _GNU_SOURCE
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../st.h"

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #condition); \
        return 1; \
    } \
} while (0)

static void write_snap(const char *dir) {
    char path[512];
    snprintf(path, sizeof(path), "%s/model.safetensors", dir);
    const char *hdr =
        "{\"weight\":{\"dtype\":\"U8\",\"shape\":[17],\"data_offsets\":[0,17]},"
        "\"weight.qs\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[17,25]}}";
    uint64_t hlen = strlen(hdr);
    unsigned char weight[17];
    float scales[2] = {0.25f, 1.5f};
    for (int i = 0; i < 17; i++) weight[i] = (unsigned char)(i * 11 + 5);
    FILE *f = fopen(path, "wb");
    fwrite(&hlen, 8, 1, f);
    fwrite(hdr, 1, hlen, f);
    fwrite(weight, 1, sizeof(weight), f);
    fwrite(scales, 1, sizeof(scales), f);
    fclose(f);
}

static void close_fixture_shards(shards *S) {
    st_mirror_reset(S);
    for (int i = 0; i < S->nfd; i++) {
        if (S->dfds[i] >= 0 && S->dfds[i] != S->fds[i]) close(S->dfds[i]);
        if (S->fds[i] >= 0) close(S->fds[i]);
        S->dfds[i] = S->fds[i] = -1;
    }
}

int main(void) {
    char dir[] = "test_st_map_XXXXXX";
    CHECK(mkdtemp(dir) != NULL);
    write_snap(dir);

    shards S;
    st_init(&S, dir);

    st_mapped_raw weight = {0}, scales = {0};
    CHECK(st_map_raw(&S, "weight", &weight) == 0);
    CHECK(weight.nbytes == 17);
    const unsigned char *w = (const unsigned char*)weight.data;
    for (int i = 0; i < 17; i++) CHECK(w[i] == (unsigned char)(i * 11 + 5));

    /* The second tensor begins at an unaligned file offset.  The mapping
     * primitive must align the OS view while returning the exact tensor byte. */
    CHECK(st_map_raw(&S, "weight.qs", &scales) == 0);
    CHECK(scales.nbytes == 8);
    float s[2];
    memcpy(s, scales.data, sizeof(s));
    CHECK(s[0] == 0.25f && s[1] == 1.5f);

    st_unmap_raw(&scales);
    st_unmap_raw(&weight);
    CHECK(scales.data == NULL && scales.nbytes == 0);
    CHECK(weight.data == NULL && weight.nbytes == 0);

    close_fixture_shards(&S);
    char path[512];
    snprintf(path, sizeof(path), "%s/model.safetensors", dir);
    CHECK(remove(path) == 0);
#ifdef _WIN32
    CHECK(_rmdir(dir) == 0);
#else
    CHECK(rmdir(dir) == 0);
#endif
    printf("test_st_map: exact unaligned read-only tensor views: ok\n");
    return 0;
}
