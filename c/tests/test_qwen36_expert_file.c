/* Real files and real reader: payload identity, structural rejection and
 * destination bounds. This fixture does not emulate qef_open or qef_read. */
#include "../qwen36_expert_file.h"
#include <assert.h>
#include <math.h>
#include <stdlib.h>

static const QefGeometry geometry = {2, 3, 64, 64, 64};
static const uint64_t fingerprint = UINT64_C(0x0123456789abcdef);

static void rejected_layout(QefGeometry g) {
    QefHeader h, before;
    char error[160];
    memset(&h, 0x5a, sizeof(h));
    before = h;
    assert(!qef_layout(&g, fingerprint, &h, error, sizeof(error)));
    assert(error[0] && !memcmp(&h, &before, sizeof(h)));
}

static void layout_tests(void) {
    QefHeader h;
    char error[160];
    assert(qef_layout(&geometry, fingerprint, &h, error, sizeof(error)));
    assert(!error[0] && sizeof(h) == 128);
    assert(h.count == 6 && h.packed_bytes == 6144 && h.scale_count == 192);
    assert(h.record_bytes == 6912 && h.stride == 16384);
    assert(h.data_offset == 16384 && h.bytes == 114688);
    assert(!h.reserved0 && !h.reserved[0] && !h.reserved[1] && !h.reserved[2]);
    assert(!qef_layout(NULL, fingerprint, &h, error, sizeof(error)));
    assert(!qef_layout(&geometry, fingerprint, NULL, NULL, 0));
    rejected_layout((QefGeometry){0, 3, 64, 64, 64});
    rejected_layout((QefGeometry){2, 0, 64, 64, 64});
    rejected_layout((QefGeometry){2, 3, 0, 64, 64});
    rejected_layout((QefGeometry){2, 3, 64, 0, 64});
    rejected_layout((QefGeometry){2, 3, 64, 64, 0});
    rejected_layout((QefGeometry){2, 3, 64, 64, 32});
    rejected_layout((QefGeometry){2, 3, 65, 64, 64});
    rejected_layout((QefGeometry){2, 3, 64, 65, 64});
    rejected_layout((QefGeometry){UINT32_MAX, UINT32_MAX, 64, 64, 64});
    rejected_layout((QefGeometry){UINT32_C(2147483648), 1, 64, 64, 64});
    /* Aligned dimensions whose three-matrix element count overflows uint64. */
    rejected_layout((QefGeometry){1, 1, UINT32_C(4294967232),
                                  UINT32_C(4294967232), 64});
    /* Record geometry fits, but total bytes exceed signed offsets. */
    rejected_layout((QefGeometry){2, 1, UINT32_C(2147483648),
                                  UINT32_C(2147483648), 64});
    /* Multiplying a valid stride by the count overflows uint64. */
    rejected_layout((QefGeometry){16, 1, UINT32_C(1073741824),
                                   UINT32_C(1073741824), 64});
    if (SIZE_MAX > UINT32_MAX) {
        QefGeometry real = {40, 256, 2048, 512, 64};
        assert(qef_layout(&real, fingerprint, &h, NULL, 0));
        assert(h.record_bytes == UINT64_C(1769472));
        assert(h.stride == h.record_bytes && h.count == 10240);
        assert(h.bytes == UINT64_C(18119409664));
        real = (QefGeometry){INT32_MAX, 1, 64, 64, 64};
        assert(qef_layout(&real, fingerprint, &h, NULL, 0));
        assert(h.count == INT32_MAX);
    }
}

/* Layout 2: a scale's code and back, the same F32 bits. */
static void code_tests(void) {
    uint16_t code;
    unsigned checked = 0;
    /* Every bfloat16 numerator from 2^-120 to 2^126: b / 7 is a normal F32. */
    for (uint32_t c = 0x0380; c < 0x7f00; c++) {
        float s = qef_code_scale((uint16_t)c);
        assert(qef_scale_code(s, &code) && code == c);
        uint32_t a, b;
        float back = qef_code_scale(code);
        memcpy(&a, &s, 4);
        memcpy(&b, &back, 4);
        assert(a == b);
        checked++;
    }
    assert(checked > 30000);
    assert(qef_scale_code(qef_eps_scale(), &code) && code == 0);
    assert(qef_code_scale(0) == qef_eps_scale());
    /* Scales that are not a bfloat16 / 7 have no code. */
    assert(!qef_scale_code(0.1f, &code));
    assert(!qef_scale_code(0.0f, &code));
    assert(!qef_scale_code(1.0f / 3.0f, &code));
    assert(!qef_scale_code(NAN, &code));
    assert(!qef_scale_code(INFINITY, &code));
}

static void codes_layout_tests(void) {
    QefHeader h;
    char error[160];
    assert(qef_layout_as(&geometry, fingerprint, QEF_LAYOUT_CODES, &h, error, sizeof(error)));
    assert(h.layout == QEF_LAYOUT_CODES && h.record_bytes == 6144 + 192 * 2 && h.stride == 16384);
    assert(qef_payload_bytes(&h) == 6912);
    assert(!qef_layout_as(&geometry, fingerprint, 3, &h, error, sizeof(error)) && error[0]);
    /* The checkpoint quantizer's convention, pinned to known values: a scale
     * is a bfloat16 max|w| over 7, and the all-zero group's is 1e-12 / 7. The
     * writer and reader only agree with each other otherwise. */
    uint16_t code;
    assert(qef_scale_code(0.5f / 7.0f, &code) && code == 0x3f00 && qef_code_scale(0x3f00) == 0.5f / 7.0f);
    assert(qef_scale_code(1e-12f / 7.0f, &code) && code == 0 && qef_code_scale(0) == 1e-12f / 7.0f);
    assert(!qef_scale_code(0.5f / 6.0f, &code));   /* not a bfloat16 over 7: the file keeps F32 */
    if (SIZE_MAX > UINT32_MAX) {
        QefGeometry real = {40, 256, 2048, 512, 64};
        assert(qef_layout_as(&real, fingerprint, QEF_LAYOUT_CODES, &h, NULL, 0));
        assert(h.record_bytes == UINT64_C(1671168) && h.stride == h.record_bytes);
        assert(h.bytes == UINT64_C(16384) + UINT64_C(10240) * UINT64_C(1671168));
        assert(qef_payload_bytes(&h) == UINT64_C(1769472));
    }
}

#ifndef _WIN32
static void write_at(int fd, const void *source, size_t bytes, uint64_t offset) {
    size_t done = 0;
    while (done < bytes) {
        ssize_t n = pwrite(fd, (const unsigned char *)source + done,
                           bytes - done, (off_t)(offset + done));
        if (n < 0 && errno == EINTR) continue;
        assert(n > 0);
        done += (size_t)n;
    }
}

static void pattern(const QefHeader *h, uint64_t key, unsigned char *out) {
    for (uint64_t i = 0; i < h->packed_bytes; ++i)
        out[i] = (unsigned char)(key * 43 + i * 17 + i / 251);
    for (uint64_t i = 0; i < h->scale_count; ++i) {
        float scale = (float)(key + 1) + (float)(i % 64) / 128.0f;
        memcpy(out + h->packed_bytes + i * 4, &scale, sizeof(scale));
    }
}

static void create_fixture(int fd, const QefHeader *h) {
    unsigned char *payload = malloc((size_t)h->record_bytes);
    assert(payload);
    assert(!ftruncate(fd, 0));
    assert(!ftruncate(fd, (off_t)h->bytes));
    write_at(fd, h, sizeof(*h), 0);
    for (uint64_t key = 0; key < h->count; ++key) {
        pattern(h, key, payload);
        write_at(fd, payload, (size_t)h->record_bytes,
                 h->data_offset + key * h->stride);
    }
    free(payload);
}

static void rejected_header(int fd, const char *path, const QefHeader *bad) {
    QefFile file = {0};
    char error[160] = {0};
    write_at(fd, bad, sizeof(*bad), 0);
    assert(!qef_open(&file, path, &geometry, fingerprint, error, sizeof(error)));
    assert(error[0] && !file.opened);
    qef_close(&file);
}

static void header_tests(int fd, const char *path, const QefHeader *h) {
    QefHeader bad;
    QefFile file = {0};
    char error[160];
#define REJECT_FIELD(field) do { bad = *h; bad.field ^= 1; \
    rejected_header(fd, path, &bad); } while (0)
    REJECT_FIELD(magic[0]);
    REJECT_FIELD(magic[7]);
    REJECT_FIELD(version);
    REJECT_FIELD(layout);
    REJECT_FIELD(layers);
    REJECT_FIELD(experts);
    REJECT_FIELD(hidden);
    REJECT_FIELD(inter);
    REJECT_FIELD(group);
    REJECT_FIELD(reserved0);
    REJECT_FIELD(fingerprint);
    REJECT_FIELD(count);
    REJECT_FIELD(packed_bytes);
    REJECT_FIELD(scale_count);
    REJECT_FIELD(record_bytes);
    REJECT_FIELD(stride);
    REJECT_FIELD(data_offset);
    REJECT_FIELD(bytes);
    REJECT_FIELD(reserved[0]);
    REJECT_FIELD(reserved[1]);
    REJECT_FIELD(reserved[2]);
#undef REJECT_FIELD
    bad = *h;
    bad.data_offset = UINT64_MAX;
    rejected_header(fd, path, &bad);
    write_at(fd, h, sizeof(*h), 0);
    assert(!qef_open(&file, path, &geometry, fingerprint ^ 1, error, sizeof(error)));
    QefGeometry other = geometry;
    other.hidden *= 2;
    assert(!qef_open(&file, path, &other, fingerprint, error, sizeof(error)));
    assert(!qef_open(&file, NULL, &geometry, fingerprint, NULL, 0));
    assert(!qef_open(NULL, path, &geometry, fingerprint, NULL, 0));
    assert(!qef_open(&file, path, NULL, fingerprint, NULL, 0));
    /* A missing path reports an open failure without taking ownership. */
    assert(!qef_open(&file, "", &geometry, fingerprint, error, sizeof(error)));
    assert(error[0] && !file.opened);
    /* A one-byte buffer gets the empty string; the size is volatile, or gcc
     * warns about the very truncation this checks. */
    char tiny[2] = {'x', 'y'};
    volatile size_t one = 1;
    assert(!qef_open(&file, NULL, &geometry, fingerprint, tiny, one));
    assert(tiny[0] == '\0' && tiny[1] == 'y');
    qef_close(&file);
}

static void read_tests(int fd, const char *path, const QefHeader *h) {
    QefFile file = {0}, unopened = {0};
    char error[160];
    size_t bytes = (size_t)h->record_bytes;
    unsigned char *expected = malloc(bytes);
    unsigned char *guarded = malloc(bytes + 2);
    assert(expected && guarded);
    /* The opened flag, not an fd value (including fd 0), controls ownership. */
    QefFile borrowed = {.fd = fd};
    qef_close(&borrowed);
    assert(fcntl(fd, F_GETFD) >= 0);
    qef_close(&unopened);
    qef_close(&unopened);
    qef_close(NULL);
    assert(qef_open(&file, path, &geometry, fingerprint, error, sizeof(error)));
    int owned_fd = file.fd;
    assert(!qef_open(&file, path, &geometry, fingerprint, error, sizeof(error)));
    assert(file.opened && file.fd == owned_fd);
    /* Reverse traversal verifies positioning rather than shared seek state. */
    for (uint64_t key = h->count; key-- > 0;) {
        memset(guarded, 0xa5, bytes + 2);
        pattern(h, key, expected);
        assert(qef_read(&file, (uint32_t)(key / h->experts),
                        (uint32_t)(key % h->experts), guarded + 1,
                        bytes, error, sizeof(error)));
        assert(!error[0] && !memcmp(guarded + 1, expected, bytes));
        assert(guarded[0] == 0xa5 && guarded[bytes + 1] == 0xa5);
    }
    memset(guarded, 0xa5, bytes + 2);
    assert(!qef_read(&file, h->layers, 0, guarded + 1, bytes, error, sizeof(error)));
    assert(!qef_read(&file, 0, h->experts, guarded + 1, bytes, NULL, 0));
    assert(!qef_read(&file, UINT32_MAX, 0, guarded + 1, bytes, NULL, 0));
    assert(!qef_read(&file, 0, UINT32_MAX, guarded + 1, bytes, NULL, 0));
    assert(!qef_read(&file, 0, 0, guarded + 1, bytes - 1, error, sizeof(error)));
    assert(!qef_read(&file, 0, 0, guarded + 1, 0, error, sizeof(error)));
    assert(!qef_read(&file, 0, 0, NULL, bytes, error, sizeof(error)));
    assert(!qef_read(&unopened, 0, 0, guarded + 1, bytes, NULL, 0));
    assert(!qef_read(NULL, 0, 0, guarded + 1, bytes, NULL, 0));
    for (size_t i = 0; i < bytes + 2; ++i) assert(guarded[i] == 0xa5);
    assert(qef_read(&file, 0, 0, guarded + 1, bytes + 1, NULL, 0));
    assert(guarded[0] == 0xa5 && guarded[bytes + 1] == 0xa5);
    /* An actual short read followed by EOF must fail, even after open passed. */
    assert(!ftruncate(fd, (off_t)(h->data_offset + h->record_bytes - 1)));
    assert(!qef_read(&file, 0, 0, guarded + 1, bytes, error, sizeof(error)));
    assert(error[0] && guarded[0] == 0xa5 && guarded[bytes + 1] == 0xa5);
    qef_close(&file);
    assert(!file.opened && file.fd == -1);
    assert(fcntl(owned_fd, F_GETFD) == -1 && errno == EBADF);
    assert(!qef_read(&file, 0, 0, guarded + 1, bytes, error, sizeof(error)));
    qef_close(&file);
    free(expected);
    free(guarded);
}

static void size_tests(int fd, const char *path, const QefHeader *h) {
    const uint64_t sizes[] = {0, 127, 128, h->data_offset,
                               h->bytes - 1, h->bytes + 1};
    for (size_t i = 0; i < sizeof(sizes) / sizeof(sizes[0]); ++i) {
        QefFile file = {0};
        char error[160];
        create_fixture(fd, h);
        assert(!ftruncate(fd, (off_t)sizes[i]));
        assert(!qef_open(&file, path, &geometry, fingerprint, error, sizeof(error)));
        assert(error[0] && !file.opened);
        qef_close(&file);
    }
}

/* A layout-2 file: codes on disk, the F32 scales they stand for in memory. */
static void codes_pattern(const QefHeader *h, uint64_t key, unsigned char *disk, unsigned char *memory) {
    for (uint64_t i = 0; i < h->packed_bytes; ++i)
        disk[i] = memory[i] = (unsigned char)(key * 43 + i * 17 + i / 251);
    for (uint64_t i = 0; i < h->scale_count; ++i) {
        uint16_t code = i % 61 == 7 ? 0 : (uint16_t)(0x3c00 + (key * 131 + i * 7) % 0x0700);
        float scale = qef_code_scale(code);
        memcpy(disk + h->packed_bytes + 2 * i, &code, 2);
        memcpy(memory + h->packed_bytes + 4 * i, &scale, 4);
    }
}

/* Run for the fixture's geometry, whose two layouts have the same size, and for
 * one expert of the 35B model's shape, where they differ (1,785,856 and
 * 1,687,552 bytes): the size check must follow the header's layout. */
static void codes_file_tests(const QefGeometry *g) {
    char path[] = "/tmp/qwen36-expert-codes-XXXXXX";
    QefHeader h;
    QefFile file = {0};
    char error[160];
    assert(qef_layout_as(g, fingerprint, QEF_LAYOUT_CODES, &h, NULL, 0));
    size_t payload = (size_t)qef_payload_bytes(&h);
    unsigned char *disk = malloc((size_t)h.record_bytes), *expected = malloc(payload), *guarded = malloc(payload + 2);
    assert(disk && expected && guarded);
    int fd = mkstemp(path);
    assert(fd >= 0 && !ftruncate(fd, (off_t)h.bytes));
    write_at(fd, &h, sizeof(h), 0);
    for (uint64_t key = 0; key < h.count; ++key) {
        codes_pattern(&h, key, disk, expected);
        write_at(fd, disk, (size_t)h.record_bytes, h.data_offset + key * h.stride);
    }
    assert(qef_open(&file, path, g, fingerprint, error, sizeof(error)));
    assert(file.h.layout == QEF_LAYOUT_CODES && file.scales);
    for (uint64_t key = h.count; key-- > 0;) {
        memset(guarded, 0xa5, payload + 2);
        codes_pattern(&h, key, disk, expected);
        assert(qef_read(&file, (uint32_t)(key / h.experts), (uint32_t)(key % h.experts), guarded + 1,
                        payload, error, sizeof(error)));
        assert(!memcmp(guarded + 1, expected, payload));
        assert(guarded[0] == 0xa5 && guarded[payload + 1] == 0xa5);
    }
    /* The destination must hold the F32 scales, not just the record. */
    assert(!qef_read(&file, 0, 0, guarded + 1, payload - 1, error, sizeof(error)) && error[0]);
    assert(!qef_read(&file, 0, 0, guarded + 1, (size_t)h.record_bytes, error, sizeof(error)));
    qef_close(&file);
    assert(!file.scales && !file.opened);
    /* A layout-2 header must match its own file size. */
    assert(!ftruncate(fd, (off_t)(h.bytes + 1)));
    assert(!qef_open(&file, path, g, fingerprint, error, sizeof(error)) && !file.opened);
    qef_close(&file);
    assert(!close(fd) && !unlink(path));
    free(disk);
    free(expected);
    free(guarded);
}

#endif /* POSIX file fixtures */

int main(void) {
    layout_tests();
    code_tests();
    codes_layout_tests();
#ifndef _WIN32
    static const QefGeometry expert_35b = {1, 1, 2048, 512, 64};
    QefHeader one, two;
    assert(qef_layout(&expert_35b, fingerprint, &one, NULL, 0) &&
           qef_layout_as(&expert_35b, fingerprint, QEF_LAYOUT_CODES, &two, NULL, 0));
    assert(one.bytes == 1785856 && two.bytes == 1687552);
    codes_file_tests(&geometry);
    codes_file_tests(&expert_35b);
    char path[] = "/tmp/qwen36-expert-file-XXXXXX";
    QefHeader h;
    assert(qef_layout(&geometry, fingerprint, &h, NULL, 0));
    int fd = mkstemp(path);
    assert(fd >= 0);
    create_fixture(fd, &h);
    header_tests(fd, path, &h);
    read_tests(fd, path, &h);
    size_tests(fd, path, &h);
    assert(!close(fd));
    assert(!unlink(path));
    puts("expert file: real payload reads, header/identity/size validation, capacity and overflow checks, "
         "16-bit scale codes passed");
#else
    puts("expert file: layout checks passed; POSIX file fixture skipped on Windows");
#endif
    return 0;
}
